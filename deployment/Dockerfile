FROM python:3.9-slim-buster

# Define the automatic platform ARGs
ARG TARGETARCH

VOLUME /app/dataset
VOLUME /app/logs

WORKDIR /app

COPY interface/requirements.txt /app/requirements_interface.txt
COPY requirements.txt /app/requirements_clientML.txt
COPY ../scripts/ld_preload_specific.sh /app
COPY ./common /app/common

# Make the scripts executable and run them based on architecture
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "Installing ARM-specific dependencies"; \
        chmod +x /app/common/install_dependencies_arm64.sh; \
        /app/common/install_dependencies_arm64.sh; \
        echo "Export ld_preload for ARM"; \
        chmod +x /app/ld_preload_specific.sh; \
        /app/ld_preload_specific.sh; \
    # elif [ "$TARGETARCH" = "amd64" ]; then \
    else \
        echo "Not installing dependencies"; \
    fi

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements_clientML.txt
RUN pip3 install -r requirements_interface.txt

COPY interface/interface.py /app
COPY ../cmd/clientML.py /app

EXPOSE 5000
CMD ["uvicorn", "interface:mflinterface_app", "--host", "0.0.0.0", "--port", "5000"]
