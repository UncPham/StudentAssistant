FROM node:22-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . . 

RUN npm run build

EXPOSE 5137

CMD [ "npm", "run", "start" ]
