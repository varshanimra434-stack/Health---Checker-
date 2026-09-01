# Auto Product Health Checker

Hindi camera-based product-label screening app built with Streamlit and Gemini.

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set the API key without putting it in `app.py`:

   ```bash
   export GEMINI_API_KEY="your-key"
   ```

   Or copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml` and fill in the key.

3. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

The app returns `SAFE`, `HARMFUL`, or `UNCERTAIN`. `UNCERTAIN` is used when
the label cannot be read clearly, so an unclear photo is not treated as safe.