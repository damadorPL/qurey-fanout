import json
from typing import List, Optional, Tuple

import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Gemini Query Fanout", layout="centered")

# ---------------------------- Utils ----------------------------


def custom_cosine_similarity(a: List[float], b: List[float]) -> float:
    try:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = dot_product / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))
    except Exception:
        return 0.0


def calculate_cosine_similarities(
    query_embedding: List[float], question_embeddings: List[List[float]]
) -> Optional[np.ndarray]:
    try:
        similarities = [
            custom_cosine_similarity(query_embedding, q) for q in question_embeddings
        ]
        arr = np.array(similarities)
        return arr
    except Exception:
        return None


# ---------------------------- Gemini APIs ----------------------------
@st.cache_data(show_spinner=False)
def get_gemini_embeddings(
    texts: Tuple[str, ...], api_key: str
) -> Optional[List[List[float]]]:
    """Get embeddings from Google Gemini embedding API for a tuple of texts.
    Uses caching to avoid repeated calls for the same inputs.
    """
    embeddings: List[List[float]] = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    for i, text in enumerate(texts):
        data = {"model": "gemini-embedding-001", "content": {"parts": [{"text": text}]}}
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            if "embedding" in result and "values" in result["embedding"]:
                embeddings.append(result["embedding"]["values"])
            else:
                st.error(f"Invalid embedding response structure for text #{i+1}")
                return None
        except Exception as e:
            st.error(f"Error getting embedding for text #{i+1}: {e}")
            return None

    if len(embeddings) == len(texts):
        return embeddings
    else:
        return None


@st.cache_data(show_spinner=False)
def query_fanout_and_rank(
    keyword: str, language: str, top_n: int, api_key: str
) -> Optional[Tuple[List[str], List[float], Optional[str]]]:
    """Generate 10 questions using Gemini generate API and rank them by cosine similarity.
    Returns top_n questions plus their similarity scores and the raw text response (when available).
    """
    # Build system and user prompts (kept similar to notebook)
    system_prompt = (
        'You are an advanced AI search assistant. Your task is to use the "query fan-out" technique '
        "to anticipate a user's complete informational need from a single keyword.\n\n"
        "You will generate a list of 10 highly semantically related and comprehensive short questions that a user might have based on this keyword.\n\n"
        'After completing your analysis, provide your output in a structured JSON format. The JSON should contain an array named "questions" with 10 string elements. The final output should contain only the JSON structure.'
    )

    user_prompt = f"The keyword you will analyze is:\n<keyword>\n{keyword}\n</keyword>\n\nLanguage of keyword, operation and output language:\n<language>\n{language}\n</language>"

    payload = {
        "contents": [{"parts": [{"text": system_prompt}, {"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        },
    }

    gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    raw_text: Optional[str] = None

    try:
        response = requests.post(gen_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "candidates" in result and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content", {})
            if "parts" in content and len(content["parts"]) > 0:
                text_response = content["parts"][0].get("text", "").strip()
                raw_text = text_response

                # Trim code fence markers if present
                if text_response.startswith("```json"):
                    text_response = text_response[7:]
                if text_response.startswith("```"):
                    text_response = text_response[3:]
                if text_response.endswith("```"):
                    text_response = text_response[:-3]
                text_response = text_response.strip()

                try:
                    questions_data = json.loads(text_response)
                    if "questions" in questions_data:
                        questions = questions_data["questions"]
                        # Get embeddings for keyword + questions
                        all_texts = tuple([keyword] + questions)
                        embeddings = get_gemini_embeddings(all_texts, api_key)

                        if embeddings and len(embeddings) == len(all_texts):
                            query_embedding = embeddings[0]
                            question_embeddings = embeddings[1:]
                            similarities = calculate_cosine_similarities(
                                query_embedding, question_embeddings
                            )
                            if similarities is not None:
                                sim_results = [
                                    (i, float(similarities[i]), questions[i])
                                    for i in range(len(questions))
                                ]
                                sim_results.sort(key=lambda x: x[1], reverse=True)
                                top = sim_results[:top_n]
                                top_questions = [t[2] for t in top]
                                top_scores = [t[1] for t in top]
                                return top_questions, top_scores, raw_text
                            else:
                                st.warning(
                                    "Failed to compute similarities, returning unranked top questions"
                                )
                                return questions[:top_n], [None] * top_n, raw_text
                        else:
                            st.warning(
                                "Failed to get embeddings, returning unranked top questions"
                            )
                            return questions[:top_n], [None] * top_n, raw_text
                    else:
                        st.error("Response JSON does not contain 'questions' field.")
                        return None
                except json.JSONDecodeError as e:
                    st.error(f"Failed parsing JSON from model: {e}")
                    return None

        st.error("Invalid response structure from Gemini generate API")
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ---------------------------- Streamlit UI ----------------------------


def main():
    st.title("🤖 Gemini Query Fanout")
    st.markdown(
        "Generate 10 semantically related questions from a keyword using Gemini and rank them using embeddings."
    )

    with st.form(key="fanout_form"):
        keyword = st.text_input(
            "Keyword", value="", placeholder="Enter a keyword to analyze"
        )
        language = st.text_input(
            "Language", value="English", placeholder="e.g., English, Polish"
        )
        top_n = st.slider("Top N results", 1, 10, 5)
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Gemini API key (kept only in browser session)",
        )
        show_raw = st.checkbox("Show raw model text response")
        submit = st.form_submit_button("Generate")

    if submit:
        if not keyword:
            st.error("Keyword cannot be empty")
            return
        if not api_key:
            st.error("API Key is required")
            return

        with st.spinner("Generating questions and computing similarities..."):
            result = query_fanout_and_rank(keyword, language, top_n, api_key)

        if result:
            questions, scores, raw_text = result
            st.success(f"Top {len(questions)} questions generated")

            for i, (q, s) in enumerate(zip(questions, scores), start=1):
                if s is not None:
                    st.write(f"**{i}.** [{s:.3f}] {q}")
                else:
                    st.write(f"**{i}.** {q}")

            if show_raw and raw_text:
                with st.expander("Raw model text"):
                    st.code(raw_text)

            output_json = json.dumps(
                {"questions": questions, "similarities": scores, "raw_text": raw_text},
                ensure_ascii=False,
                indent=2,
            )

            st.download_button(
                "Download JSON",
                data=output_json,
                file_name=f"query_fanout_{keyword}.json",
                mime="application/json",
            )

    st.markdown("---")
    st.markdown(
        "**Notes:**\n- The API key is sent to Google's Gemini API endpoints.\n- Use the `Download JSON` button to save results locally."
    )


if __name__ == "__main__":
    main()
