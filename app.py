import streamlit as st
import numpy as np
from dotenv import load_dotenv

from utils.pdf_processing import load_pdf, extract_toc
from utils.chunking import create_chunks
from utils.embeddings import generate_embeddings
from utils.vector_store import create_faiss_index
from utils.retrieval import retrieve_context
from utils.llm import generate_answer, generate_summary, generate_flashcards, generate_quiz

load_dotenv()

st.set_page_config(
    page_title="AI Study Copilot",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def get_chapter_index(chapter, chunks, embeddings):
    """
    What enters : chapter dict (title, start_page, end_page),
                  all chunks (list of Documents),
                  all embeddings (numpy array aligned with chunks by index)
    What leaves : (chapter_chunks, chapter_index) or (None, None) if empty
    Why         : chapter-scoped retrieval needs a small FAISS index built
                  only from chunks that belong to the selected chapter's
                  page range — avoids returning results from other chapters
    """

    # TOC pages are 1-indexed; chunk metadata["page"] is 0-indexed
    start = chapter["start_page"] - 1
    end   = chapter["end_page"]   - 1

    chapter_chunks     = []
    chapter_embeddings = []

    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        if start <= page <= end:
            chapter_chunks.append(chunk)
            chapter_embeddings.append(embeddings[i])

    if not chapter_chunks:
        return None, None

    chapter_embeddings_arr = np.array(chapter_embeddings, dtype=np.float32)
    chapter_index = create_faiss_index(chapter_embeddings_arr)

    return chapter_chunks, chapter_index


def get_full_chapter_context(chapter, chunks):
    """
    What enters : chapter dict, all document chunks
    What leaves : all chapter text joined as one string
    Why         : summary / flashcard / quiz generation needs to see the
                  whole chapter, not just the top-k retrieved chunks
    """

    start = chapter["start_page"] - 1
    end   = chapter["end_page"]   - 1

    texts = [
        chunk.page_content
        for chunk in chunks
        if start <= chunk.metadata.get("page", 0) <= end
    ]

    return "\n\n".join(texts)


def init_chapter_cache(title):
    """Create a fresh cache entry for a chapter the first time it is opened."""

    if "cache" not in st.session_state:
        st.session_state["cache"] = {}

    if title not in st.session_state["cache"]:
        st.session_state["cache"][title] = {
            "summary":           None,
            "flashcards":        None,
            "quiz":              None,
            "chat_history":      [],
            "flashcard_index":   0,
            "flashcard_flipped": False,
            "flashcard_marks":   {},
            "quiz_answers":      {},
            "quiz_submitted":    False,
        }


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:

    st.title("📚 AI Study Copilot")
    st.divider()

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:

        # Process only when a new file is uploaded
        new_file = (
            "pdf_name" not in st.session_state
            or st.session_state["pdf_name"] != uploaded_file.name
        )

        if new_file:

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing PDF..."):

                documents  = load_pdf("temp.pdf")
                chunks     = create_chunks(documents)
                embeddings = generate_embeddings(chunks)
                index      = create_faiss_index(embeddings)
                toc        = extract_toc("temp.pdf")

            st.session_state.update({
                "pdf_name":         uploaded_file.name,
                "chunks":           chunks,
                "embeddings":       np.array(embeddings, dtype=np.float32),
                "index":            index,
                "toc":              toc,
                "chapter_indexes":  {},
                "cache":            {},
                "selected_chapter": None,
            })


        # ── PDF name ──────────────────────────────
        st.markdown(f"**{st.session_state['pdf_name']}**")
        st.divider()


        # ── Chapter list ──────────────────────────
        st.markdown("**Contents**")

        cache = st.session_state.get("cache", {})

        for chapter in st.session_state["toc"]:

            title = chapter["title"]

            # Mark chapters the student has already engaged with
            studied = title in cache and any([
                cache[title].get("summary"),
                cache[title].get("flashcards"),
                cache[title].get("quiz"),
                len(cache[title].get("chat_history", [])) > 0,
            ])

            label    = f"{'✓  ' if studied else ''}{title}"
            selected = st.session_state.get("selected_chapter") == title

            if st.button(
                label,
                key=f"ch_{title}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state["selected_chapter"] = title
                st.rerun()

        # Debug: shows the raw TOC fitz extracted — remove once working
        with st.expander("Debug: raw TOC"):
            import fitz as _fitz
            _doc = _fitz.open("temp.pdf")
            st.write(_doc.get_toc())
            _doc.close()


# ─────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────

if "pdf_name" not in st.session_state:

    st.title("📚 AI Study Copilot")
    st.markdown("Upload a PDF in the sidebar to begin.")

elif st.session_state.get("selected_chapter") is None:

    st.title("📚 AI Study Copilot")
    st.markdown("Select a chapter from the sidebar.")

else:

    chapter_title = st.session_state["selected_chapter"]
    toc           = st.session_state["toc"]
    chapter       = next(c for c in toc if c["title"] == chapter_title)

    init_chapter_cache(chapter_title)
    cache = st.session_state["cache"][chapter_title]


    # Build a chapter-specific FAISS index the first time this chapter is opened.
    # We slice the stored embeddings array by page range — no recomputation needed.
    if chapter_title not in st.session_state["chapter_indexes"]:

        chapter_chunks, chapter_index = get_chapter_index(
            chapter,
            st.session_state["chunks"],
            st.session_state["embeddings"],
        )
        st.session_state["chapter_indexes"][chapter_title] = (chapter_chunks, chapter_index)

    else:

        chapter_chunks, chapter_index = st.session_state["chapter_indexes"][chapter_title]


    # ── Chapter header ────────────────────────
    st.header(chapter_title)
    st.caption(f"Pages {chapter['start_page']} – {chapter['end_page']}")

    tab_chat, tab_summary, tab_flashcards, tab_quiz = st.tabs([
        "💬 Chat", "📝 Summary", "🃏 Flashcards", "🧠 Quiz",
    ])


    # ─────────────────────────────────────────────────────────────
    # CHAT TAB
    # ─────────────────────────────────────────────────────────────

    with tab_chat:

        if chapter_chunks is None:
            st.warning("No content found for this chapter.")

        else:

            # Render conversation history
            for entry in cache["chat_history"]:

                with st.chat_message("user"):
                    st.write(entry["question"])

                with st.chat_message("assistant"):
                    st.write(entry["answer"])
                    if entry["source_pages"]:
                        st.caption(
                            "Sources: page(s) "
                            + ", ".join(str(p) for p in entry["source_pages"])
                        )

            # Input form — form prevents reruns on every keystroke
            with st.form("chat_form"):
                question = st.text_input("Ask a question about this chapter")
                ask      = st.form_submit_button("Ask")

            if ask and question:

                with st.spinner("Thinking..."):

                    # Embed query → search chapter index → retrieve top-k chunks
                    result = retrieve_context(question, chapter_index, chapter_chunks)

                    # Send retrieved context + question to the LLM
                    answer = generate_answer(question, result["context"])

                cache["chat_history"].append({
                    "question":    question,
                    "answer":      answer,
                    "source_pages": result["source_pages"],
                })

                st.rerun()


    # ─────────────────────────────────────────────────────────────
    # SUMMARY TAB
    # ─────────────────────────────────────────────────────────────

    with tab_summary:

        if cache["summary"] is None:

            if st.button("Generate Summary", type="primary", key="gen_summary"):

                with st.spinner("Generating summary..."):

                    # Send the whole chapter to the LLM (not just retrieved chunks)
                    context = get_full_chapter_context(chapter, st.session_state["chunks"])
                    context = context[:20000]

                    cache["summary"] = generate_summary(context, chapter_title)

                st.rerun()

        else:

            st.markdown(cache["summary"])
            st.divider()

            if st.button("Regenerate", key="regen_summary"):
                cache["summary"] = None
                st.rerun()


    # ─────────────────────────────────────────────────────────────
    # FLASHCARDS TAB
    # ─────────────────────────────────────────────────────────────

    with tab_flashcards:

        if cache["flashcards"] is None:

            if st.button("Generate Flashcards", type="primary", key="gen_fc"):

                with st.spinner("Generating flashcards..."):

                    context = get_full_chapter_context(chapter, st.session_state["chunks"])
                    context = context[:20000]
                    cards   = generate_flashcards(context, chapter_title)

                cache["flashcards"]        = cards
                cache["flashcard_index"]   = 0
                cache["flashcard_flipped"] = False
                cache["flashcard_marks"]   = {}

                st.rerun()

        else:

            cards   = cache["flashcards"]
            idx     = cache["flashcard_index"]
            flipped = cache["flashcard_flipped"]
            marks   = cache["flashcard_marks"]
            total   = len(cards)

            if not cards:
                st.warning("No flashcards were generated.")

            else:

                card = cards[idx]

                # Progress bar
                known_count  = sum(1 for v in marks.values() if v == "known")
                review_count = sum(1 for v in marks.values() if v == "review")

                st.markdown(f"Card **{idx + 1}** of **{total}**")
                st.progress(
                    known_count / total,
                    text=f"{known_count} known · {review_count} to review",
                )

                # Card face — question or answer
                with st.container(border=True):

                    if not flipped:
                        st.markdown("**Question**")
                        st.markdown(f"#### {card['question']}")
                        if st.button("Show Answer", key=f"fc_show_{idx}", use_container_width=True):
                            cache["flashcard_flipped"] = True
                            st.rerun()

                    else:
                        st.markdown("**Answer**")
                        st.markdown(f"#### {card['answer']}")
                        if st.button("Hide Answer", key=f"fc_hide_{idx}", use_container_width=True):
                            cache["flashcard_flipped"] = False
                            st.rerun()

                # Navigation and marking
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    if st.button(
                        "← Prev",
                        key="fc_prev",
                        disabled=(idx == 0),
                        use_container_width=True,
                    ):
                        cache["flashcard_index"]   -= 1
                        cache["flashcard_flipped"]  = False
                        st.rerun()

                with c2:
                    if st.button(
                        "Next →",
                        key="fc_next",
                        disabled=(idx == total - 1),
                        use_container_width=True,
                    ):
                        cache["flashcard_index"]   += 1
                        cache["flashcard_flipped"]  = False
                        st.rerun()

                with c3:
                    mark = marks.get(idx)
                    if st.button(
                        "✓ Known",
                        key="fc_known",
                        use_container_width=True,
                        type="primary" if mark == "known" else "secondary",
                    ):
                        cache["flashcard_marks"][idx] = "known"
                        st.rerun()

                with c4:
                    if st.button(
                        "⚑ Review",
                        key="fc_review",
                        use_container_width=True,
                        type="primary" if mark == "review" else "secondary",
                    ):
                        cache["flashcard_marks"][idx] = "review"
                        st.rerun()

                st.divider()

                if st.button("Regenerate Flashcards", key="regen_fc"):
                    cache["flashcards"] = None
                    st.rerun()


    # ─────────────────────────────────────────────────────────────
    # QUIZ TAB
    # ─────────────────────────────────────────────────────────────

    with tab_quiz:

        if cache["quiz"] is None:

            if st.button("Generate Quiz", type="primary", key="gen_quiz"):

                with st.spinner("Generating quiz..."):

                    context   = get_full_chapter_context(chapter, st.session_state["chunks"])
                    context   = context[:20000]
                    questions = generate_quiz(context, chapter_title)

                cache["quiz"]           = questions
                cache["quiz_answers"]   = {}
                cache["quiz_submitted"] = False

                st.rerun()

        else:

            questions = cache["quiz"]

            if not questions:
                st.warning("No quiz questions were generated.")

            elif not cache["quiz_submitted"]:

                st.markdown("Answer all questions and click **Submit**.")
                st.divider()

                # Show all questions on one page
                for i, q in enumerate(questions):
                    st.markdown(f"**{i + 1}. {q['question']}**")
                    st.radio(
                        "",
                        q["options"],
                        key=f"quiz_{chapter_title}_{i}",
                        index=None,
                        label_visibility="collapsed",
                    )
                    st.write("")

                # Count how many have been answered (read from widget state)
                answered = sum(
                    1 for i in range(len(questions))
                    if st.session_state.get(f"quiz_{chapter_title}_{i}") is not None
                )

                st.divider()
                st.caption(f"{answered} of {len(questions)} answered")

                if st.button(
                    "Submit Quiz",
                    type="primary",
                    key="quiz_submit",
                    disabled=(answered < len(questions)),
                ):
                    # Collect answers from Streamlit's widget state
                    # Each radio value is the full option string e.g. "A. Paris"
                    # We store just the letter: "A"
                    answers = {}
                    for i in range(len(questions)):
                        val = st.session_state.get(f"quiz_{chapter_title}_{i}")
                        if val:
                            answers[i] = val[0]

                    cache["quiz_answers"]   = answers
                    cache["quiz_submitted"] = True
                    st.rerun()

            else:

                # ── Results ──────────────────────────────
                answers = cache["quiz_answers"]
                total   = len(questions)
                score   = sum(
                    1 for i, q in enumerate(questions)
                    if answers.get(i) == q["answer"]
                )

                if score == total:
                    st.success(f"Perfect score!  {score} / {total}")
                elif score >= total * 0.7:
                    st.success(f"Score: {score} / {total}")
                else:
                    st.warning(f"Score: {score} / {total}")

                st.progress(score / total)
                st.divider()

                # Weak topics — questions answered incorrectly
                wrong = [
                    (i, q) for i, q in enumerate(questions)
                    if answers.get(i) != q["answer"]
                ]

                if wrong:
                    st.markdown("**Topics to review:**")

                    for i, q in wrong:
                        with st.expander(q["question"]):
                            st.markdown(f"Your answer: **{answers.get(i, '—')}**")
                            st.markdown(f"Correct answer: **{q['answer']}**")
                            if q.get("explanation"):
                                st.caption(q["explanation"])
                else:
                    st.markdown("All answers correct!")

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Retake Quiz", key="quiz_retake"):
                        for i in range(total):
                            key = f"quiz_{chapter_title}_{i}"
                            if key in st.session_state:
                                del st.session_state[key]
                        cache["quiz_answers"]   = {}
                        cache["quiz_submitted"] = False
                        st.rerun()

                with col2:
                    if st.button("Regenerate Quiz", key="regen_quiz"):
                        for i in range(total):
                            key = f"quiz_{chapter_title}_{i}"
                            if key in st.session_state:
                                del st.session_state[key]
                        cache["quiz"]           = None
                        cache["quiz_answers"]   = {}
                        cache["quiz_submitted"] = False
                        st.rerun()
