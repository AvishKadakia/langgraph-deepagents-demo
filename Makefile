.PHONY: up down logs backend-shell runtime-shell frontend-shell db-shell

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

backend-shell:
	docker exec -it deepagent-backend bash

runtime-shell:
	docker exec -it deepagent-runtime sh

frontend-shell:
	docker exec -it deepagent-frontend sh

db-shell:
	docker exec -it deepagent-postgres psql -U postgres -d deepagent
