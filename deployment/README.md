
# Deployment Guide

## 1. Deploy Backend (Hugging Face Spaces)

This will host the heavy AI model.

1.  Create a new **Space** on [Hugging Face](https://huggingface.co/new-space).
    -   **Space Name**: `chess-ai-backend` (or similar)
    -   **SDK**: `Docker` (Recommended)
    -   **Visibility**: Public
2.  Upload the contents of the `deployment/huggingface` folder to the Space.
    -   `Dockerfile`
    -   `app.py`
    -   `requirements.txt`
    -   `model.pt` (Make sure this file is present!)
3.  Wait for the Space to build and run.
4.  Copy the **Space URL** (e.g., `https://username-chess-ai-backend.hf.space`).
    -   *Note: Add `/api/ai_move` to the end for testing.*

## 2. Configure Frontend

1.  Open `deployment/web_ui/index.html` in a text editor.
2.  Find line 348:
    ```javascript
    const API_URL = "https://YOUR-HF-SPACE-URL.hf.space/api/ai_move";
    ```
3.  Replace `https://YOUR-HF-SPACE-URL.hf.space` with your actual Space URL.
    -   Example: `const API_URL = "https://rhendy-chess-v27.hf.space/api/ai_move";`

## 3. Host Frontend (UI)

Now you have a simple static HTML file. You can host it anywhere for free:

*   **Netlify**: Drag and drop the `deployment/web_ui` folder.
*   **Vercel**: Install Vercel CLI or upload via Git.
*   **GitHub Pages**: Push `deployment/web_ui` to a repo and enable Pages.
*   **Directly**: Just open `index.html` in your browser (though some API calls might be blocked by local CORS policies, better to use a local server like `python -m http.server`).
