---
title: Spam Message Classifier 📧
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

## ⚙️ CI/CD Process (with GitHub Actions)

This project adopts a modern CI/CD approach to automate the software development lifecycle. Every `git push` to the `main` branch automatically triggers the following steps:

1.  **Trigger:** Code is pushed to the `main` branch on GitHub.
2.  **Workflow Start:** The GitHub Actions workflow defined in `.github/workflows/deploy.yml` starts running.
3.  **Environment Setup:**
    *   Code is checked out from the GitHub repository to the Actions Runner (`actions/checkout`), including Git LFS files (`spam_classifier.pkl`).
    *   The specified Python version (`3.10`) is set up (`actions/setup-python`).
    *   All Python dependencies listed in `requirements.txt` (Streamlit, Scikit-learn, pytest, etc.) are installed (`pip install`).
4.  **🧪 Continuous Integration (CI) - Automated Testing:**
    *   The command `pytest tests/` is executed, running all automated tests located in the `tests/` folder (e.g., model loading, basic prediction).
    *   **Critical Step:** If **any** of these tests fail, the workflow stops here, **preventing** the faulty code from being deployed.
5.  **🚀 Continuous Deployment (CD) - Deployment (If Tests Pass):**
    *   The target Hugging Face Space repository is securely cloned using the `HF_TOKEN` secret.
    *   The updated files from the GitHub repository (code, requirements, `.gitattributes`, etc.) are copied/synchronized to the cloned Space repository (`rsync`).
    *   Git LFS hooks are activated within the Space repository clone (`git lfs install --local`), and the presence of `.gitattributes` is verified.
    *   Changes are committed and pushed to the Hugging Face Space using `git push`. Git LFS automatically handles the `.pkl` file correctly.
6.  **Live Environment:** Hugging Face Space detects the newly pushed files, automatically builds the new version of the application, and makes it available to users.

Thanks to this automation, developments are delivered to the live environment quickly, reliably, and consistently.

## 🌐 Accessing the Application

You can access the live version of the application via the following Hugging Face Space link:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ridvanyigit/streamlit_hf_projem)

➡️ **[https://huggingface.co/spaces/ridvanyigit/streamlit_hf_projem](https://huggingface.co/spaces/ridvanyigit/streamlit_hf_projem)**

*(Note: It might take a few seconds for the Space to wake up from sleep.)*

## 💻 Code Repository

You can access all the source code and configuration files for the project in the following GitHub repository:

➡️ **[https://github.com/ridvanyigit/streamlit_hf_projem](https://github.com/ridvanyigit/streamlit_hf_projem)**

---