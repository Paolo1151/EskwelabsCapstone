mkdir -p ~/.streamlit/

echo "\[general]\nemail = \"email@domain\"\n" > ~/.streamlit/credentials.toml

echo "[server]\nheadless = true\n\enableCORS=false\nport = $PORT\n" > ~/.streamlit/config.toml