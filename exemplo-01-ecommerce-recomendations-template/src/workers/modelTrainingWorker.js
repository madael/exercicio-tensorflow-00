// import * as tf from '@tensorflow/tfjs';
// import * as tf from '@tensorflow/tfjs';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = {};

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

// Normaliza um valor entre 0 e 1 com base em um mínimo e máximo
const normalize = (value, min, max) => (value - min) / ((max - min) || 1);
function makeContext(users, products) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    const ageMin = Math.min(...ages);
    const ageMax = Math.max(...ages);
    const priceMin = Math.min(...prices);
    const priceMax = Math.max(...prices);

    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorIndex = Object.fromEntries(
        colors.map((color, idx) => [color, idx])
    );
    const categoryIndex = Object.fromEntries(
        categories.map((cat, idx) => [cat, idx])
    );

    const midAge = (ageMin + ageMax) / 2;
    const ageSuns = {};
    const ageCounts = {};

    // computar a media de idade dos compradores por produto
    users.forEach(user => {
        user.purchases.forEach(pid => {
            const name = pid.name;
            if (!ageSuns[name]) {
                ageSuns[name] = 0;
                ageCounts[name] = 0;
            }
            ageSuns[name] += user.age;
            ageCounts[name] += 1;
        });
    });

    const productAvg = Object.fromEntries(
        products.map(p => {
            const avgAge = ageSuns[p.name] ? ageSuns[p.name] / ageCounts[p.name] : midAge;
            return [p.name, normalize(avgAge, ageMin, ageMax)];
        })
    );

    return {
        products,
        users,
        colorIndex,
        categoryIndex,
        productAvg,
        ageMin,
        ageMax,
        priceMin,
        priceMax,
        numCategories: categories.length,
        numColors: colors.length,
        // idade + preço + categorias + cores
        dimentions: 2 + categories.length + colors.length
    }
}

const oneHotWeighed = (index, length, weight) => {
    return tf.oneHot(index, length).cast('float32').mul(weight);
}
function encodeProduct(product, context) {

    const price = tf.tensor1d([
        normalize(
            product.price,
            context.priceMin,
            context.priceMax)
        * WEIGHTS.price
    ]);

    const age = tf.tensor1d([
        (
            context.productAvg[product.name] ?? 0.5
        ) * WEIGHTS.age
    ]);

    const category = oneHotWeighed(
        context.categoryIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    );

    const color = oneHotWeighed(
        context.colorIndex[product.color],
        context.numColors,
        WEIGHTS.color
    );

    return tf.concat(
        [price, age, category, color]
    );
}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVectior = encodeUser(user, context).dataSync();
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync();

                const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;
                inputs.push([...userVectior, ...productVector]);
                labels.push(label);
            });
        });

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2
    }
}

function encodeUser(user, context) {
    if (user.purchases.length != 0) {
        return tf.stack(
            user.purchases.map(p => {
                return encodeProduct(p, context);
            })
        ).mean(0)
            .reshape([1, context.dimentions]);
    }

    return tf.concat1d([
        tf.zeros([1]),
        tf.tensor1d([
            normalize(user.age, context.ageMin, context.ageMax)
            * WEIGHTS.age
        ]),
        tf.zeros([context.numCategories]),
        tf.zeros([context.numColors])
    ]).reshape([1, context.dimentions]);
}

async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [trainData.inputDimention], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await (await fetch('/data/products.json')).json();

    const context = makeContext(users, products);

    context.productVectors = products.map(p => {
        return {
            name: p.name,
            meta: { ...p },
            vector: encodeProduct(p, context).dataSync()
        };
    });

    _globalCtx = context;

    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
function recommend(user, ctx) {
    if (!_model) { return; }

    const context = _globalCtx;


    const userVector = encodeUser(user, ctx).dataSync();
    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector];
    })
    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);

    const score = predictions.dataSync();
    const recommendations = context.productVectors
        .map((p, idx) => {
            return {
                ...p.meta,
                name: p.name,
                score: score[idx]
            }
        })
        .sort((a, b) => b.score - a.score);
    
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: recommendations
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
