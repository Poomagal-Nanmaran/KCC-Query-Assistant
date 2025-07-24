docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/vectorstore:/app/vectorstore \
  -p 8501:8501 \
  kcc-query-assistant
