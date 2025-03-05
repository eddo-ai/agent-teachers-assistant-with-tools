FROM langchain/langgraph-api:3.12



# -- Adding non-package dependency src --
ADD ./src /deps/__outer_src/src
RUN set -ex && \
    for line in '[project]' \
                'name = "src"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]'; do \
        echo "$line" >> /deps/__outer_src/pyproject.toml; \
    done
# -- End of non-package dependency src --

# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"graph": "/deps/__outer_src/src/agent_arcade_tools/graph.py:graph"}'

