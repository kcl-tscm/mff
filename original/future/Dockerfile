FROM jupyter/tensorflow-notebook

LABEL maintainer="Adam Fekete <adam.fekete@kcl.ac.uk>"
LABEL version="0.9"

# Install Ase, ASAP
RUN conda install --quiet --yes  --no-update-dependencies \
    ase asap && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER


ADD m_ff /home/jovyan/



## Final permissions fix
#USER root
#RUN sudo chown --recursive jovyan:users .
#USER jovyan