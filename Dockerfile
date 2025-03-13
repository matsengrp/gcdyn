FROM quay.io/matsengrp/partis
# FROM partis-local

ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER_GID . /partis/projects/gcdyn/
WORKDIR /partis/projects/gcdyn
RUN make install
WORKDIR /partis
# necessary for singularity:
ENV PATH /opt/conda/bin:/opt/conda/condabin:$PATH
