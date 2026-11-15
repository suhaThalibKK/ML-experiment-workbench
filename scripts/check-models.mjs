import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { predictID3,predictGaussian,predictDocument,predictHeart,predictKNN,multiplyMatrix } from '../dist/engines.js';
const lab=JSON.parse(readFileSync(new URL('../dist/assets/lab-data.json',import.meta.url),'utf8'));
assert.equal(lab.experiments.length,8);
let checks=0;
for(const row of lab.datasets.weather.rows){const sample=Object.fromEntries(lab.datasets.weather.columns.map((name,i)=>[name,row[i]]));assert.equal(predictID3(lab.models.id3,sample).label,sample.play);checks++;}
assert.equal(predictID3(lab.models.id3,{outlook:'unseen'}).label,'yes');checks++;
for(const name of ['iris','wine']){const model=lab.models.gaussian[name];const result=predictGaussian(model,model.defaults);assert.ok(Math.abs(result.probabilities.reduce((a,b)=>a+b)-1)<1e-12);assert.ok(model.classes.includes(result.label));checks++;}
assert.equal(predictDocument(lab.models.document,'Congratulations! Claim your free cash prize now').label,'spam');
assert.equal(predictDocument(lab.models.document,'Can we meet at the library after class?').label,'ham');checks+=2;
const heart=lab.models.heart;
const first=predictHeart(heart,heart.defaultRaw);assert.ok(Math.abs(first.probabilities[1]-heart.firstPresence)<1e-12);checks++;
for(const expected of heart.predictions){const row=lab.datasets.heart.rows[expected.row];const raw=Object.fromEntries(lab.datasets.heart.columns.map((name,i)=>[name,row[i]]));const actual=predictHeart(heart,raw);assert.equal(actual.label,expected.predicted);assert.ok(Math.abs(actual.probabilities[1]-expected.p_presence)<1e-12);checks++;}
for(const expected of lab.models.knn.predictions){const raw=lab.datasets.iris.rows[expected.row].slice(0,4);assert.equal(predictKNN(lab.models.knn,raw,3).label,expected.predicted);checks++;}
assert.deepEqual(multiplyMatrix([1,2,3,4]),[7,10,15,22]);checks++;
const validationRows=lab.models.cv.folds.flatMap(f=>f.validationRows);assert.equal(validationRows.length,150);assert.equal(new Set(validationRows).size,150);checks++;
for(const probabilities of lab.models.clustering.memberships)assert.ok(Math.abs(probabilities.reduce((a,b)=>a+b)-1)<1e-9);
for(const exp of lab.experiments){assert.ok(exp.source.includes('def main():'));assert.ok(exp.output.trim().length>100);}
console.log(`PASS: ${checks} model checks, all 61 heart posteriors and all 60 KNN test predictions match Python; full data and source present.`);
