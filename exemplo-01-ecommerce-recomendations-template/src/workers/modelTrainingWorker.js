// import * as tf from '@tensorflow/tfjs';
// import * as tf from '@tensorflow/tfjs';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

// Normaliza um valor entre 0 e 1 com base em um mínimo e máximo
const normalize = (value, min, max) => (value - min) / ((max - min) || 1);
function makeContext(users, catalog) {
    const ages = users.map(u => u.age);
    const prices = catalog.map(p => p.price);

    const ageMin = Math.min(...ages);
    const ageMax = Math.max(...ages);
    const priceMin = Math.min(...prices);
    const priceMax = Math.max(...prices);

    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];

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
        catalog.map(p => {
            const avgAge = ageSuns[p.name] ? ageSuns[p.name] / ageCounts[p.name] : midAge;
            return [p.name, normalize(avgAge, ageMin, ageMax)];
        })
    );

    return {
        catalog,
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
    tf.oneHot(index, length).cast('float32').mul(weight);
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
    debugger;

}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await (await fetch('/data/products.json')).json();

    const context = makeContext(users, catalog);

    context.productVectors = catalog.map(p => {
        return {
            name: p.name,
            meta: { ...p },
            vector: encodeProduct(p, context)
        };
    });



    console.log('Catalog loaded:', catalog);

    debugger;

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
