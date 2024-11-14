from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from langserve import add_routes

from app.agent.doc_agent import DOC_RAG_AGENT

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)
add_routes(
    app,
    DOC_RAG_AGENT,
    path="/doc_rag_agent",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
