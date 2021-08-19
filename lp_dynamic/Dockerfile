FROM python:3.7-slim

# copy the source code to the Docker image
COPY . /root/content/

# set our working directory to where the application code
# is located
WORKDIR /root/content/

# install the python package requirements
RUN pip install --no-cache-dir -r requirements.txt

ARG GITHUB_TOKEN
RUN pip install git+https://$GITHUB_TOKEN@https://github.com/sikauk-rad/lp_functions.git

# expose the port that our app runs on
EXPOSE 8050

