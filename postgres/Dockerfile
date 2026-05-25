FROM postgres:16-alpine

# LangGraph's PostgresSaver creates its own checkpoint tables during backend startup.
# This image exists so the checkpoint database is reproducible and easy to extend.
COPY init.sql /docker-entrypoint-initdb.d/01-init.sql
