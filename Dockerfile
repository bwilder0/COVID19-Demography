FROM python:3

RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip

# Install Demography
#RUN apt --no-cache add git
RUN git clone https://github.com/bwilder0/COVID19-Demography 
RUN cd COVID19-Demography

COPY . .

RUN /bin/bash -c "python3 -m pip install -r requirements.txt"

CMD ["python3", "run_simulation.py >> output.log"]