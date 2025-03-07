"""Script to run the Streamlit frontend."""

import os
import sys

import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Get the absolute path to the streamlit app
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "streamlit_app.py")

    # Set up Streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=localhost",
    ]

    # Run the Streamlit app
    sys.exit(stcli.main())
