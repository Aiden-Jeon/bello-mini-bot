run-litellm:
	docker compose up -d

stop-litellm:
	docker compose down -v

test-litellm:
	poetry run python test_litellm_proxy.py

run-chatbot:
	poetry run streamlit run src/app/app.py 
