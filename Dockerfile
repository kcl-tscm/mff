FROM jupyter/scipy-notebook

LABEL maintainer="Adam Fekete <adam.fekete@kcl.ac.uk>"

USER $NB_UID

# Install Python 3 packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images
RUN conda install --quiet --yes \
    'mkl-service' \
    'theano' \
    'ase' \
    'asap3' \
    'scikit-learn' \
    'nglview' && \
    conda remove --quiet --yes --force qt pyqt && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install mff from source
#RUN cd /tmp && \
#    git clone https://github.com/kcl-tscm/mff.git && \
#    cd mff && \
#    python setup.py install && \
#    cd && \
#    rm -rf /tmp/mff && \
#    fix-permissions $CONDA_DIR && \
#    fix-permissions /home/$NB_USER

COPY . /tmp/mff/
USER root
RUN chown -R $NB_UID /tmp/mff

USER $NB_UID

RUN cd /tmp/mff && \
    python setup.py install && \
    rm -rf /tmp/mff && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

WORKDIR /home/$NB_USER/
