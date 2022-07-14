FROM node:8.16.0-alpine

COPY . /PLPSLAM-viewer/

RUN set -x && \
  cd /PLPSLAM-viewer/ && \
  npm install

ENTRYPOINT ["node", "/PLPSLAM-viewer/app.js"]
