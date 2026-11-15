import assert from 'node:assert/strict';
import {categoricalNB,frequencyNB,clusterIteration,retrievalMetrics,silhouetteFor,alarmProfile,alarmJoint,kmeansPoints,silhouetteClusters} from '../dist/pyq-math.js';
import {questions,buyRows,employeeRows,fruitRows,internalFruit} from '../dist/pyq-data.js';
import {inspectGuide} from '../dist/pyq.js';
const close=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const buyQuery={Age:'youth',Income:'high',Student:'yes',Credit_rate:'excellent'};
const zero=categoricalNB(buyRows,['Age','Income','Student','Credit_rate'],'Buys_computer',buyQuery);
assert.equal(zero.winner,null);assert.equal(zero.evidence,0);assert.ok(zero.terms.every(r=>r.posterior===null));
const smooth=categoricalNB(buyRows,['Age','Income','Student','Credit_rate'],'Buys_computer',buyQuery,1);
assert.equal(smooth.winner,'yes');close(smooth.terms[0].score,(5/8)*(1/8)*(3/8)*(4/7)*(3/7));close(smooth.terms[1].score,(3/8)*(2/6)*(1/6)*(2/5)*(3/5));close(smooth.terms[0].posterior,.5893137768465164);
const employee=categoricalNB(employeeRows,['Department','Age','Salary'],'Status',{Department:'Sales',Age:'31…35',Salary:'46…50k'},0,'Count');
assert.equal(employee.total,169);assert.deepEqual(employee.terms.map(r=>r.n),[52,117]);assert.equal(employee.winner,'Senior');close(employee.terms[0].score,(52/169)*(30/52)*(35/52)*(40/52));close(employee.terms[1].score,(117/169)*(80/117)*(44/117)*(23/117));
const fruit=frequencyNB(fruitRows,['Yellow','Sweet','Long']);assert.equal(fruit.winner,'Banana');assert.equal(fruit.total,1200);close(fruit.terms[1].score,.21875);close(fruit.terms[2].score,1/108);close(fruit.terms[1].posterior,189/197);
assert.deepEqual(['Yellow','Sweet','Long'].map(f=>fruitRows.reduce((s,r)=>s+r[f],0)),[800,850,400]);assert.ok(fruitRows.every(r=>['Yellow','Sweet','Long'].every(f=>r[f]<=r.Total)));
assert.equal(frequencyNB(internalFruit,['Yellow','Sweet','Long']).winner,'Banana');
const metrics=retrievalMetrics();assert.deepEqual([metrics.tp,metrics.fp,metrics.fn,metrics.tn],[40,10,15,15]);close(metrics.precision,.8);close(metrics.recall,8/11);close(metrics.f1,16/21);close(metrics.accuracy,55/80);close(metrics.fpr,.4);assert.throws(()=>retrievalMetrics(10,8,8,0));
assert.equal(alarmJoint('unselected'),null);for(const profile of ['keepTrue','keepFalse','normalize'])for(const row of alarmProfile(profile))close(row.t+row.f,1);
close(alarmJoint('keepTrue'),.679773411135);close(alarmJoint('keepFalse'),.000680453865);close(alarmJoint('normalize'),.3402269325);assert.throws(()=>alarmProfile('textbook'));
const first=clusterIteration(kmeansPoints,[kmeansPoints[0],kmeansPoints[3],kmeansPoints[6]]);assert.deepEqual(first.labels,[0,0,0,1,0,1,2,0]);assert.deepEqual(first.next,[[4.4,7.6],[7.5,5.5],[1,2]]);
const second=clusterIteration(kmeansPoints,first.next);assert.deepEqual(second.labels,[0,0,1,0,0,1,2,0]);assert.deepEqual(second.next,[[4.6,8.4],[7,3.5],[1,2]]);
// Every supported distinct starting-point combination must retain eight points,
// use a nearest centre under the specified metric, and produce finite means.
let starts=0;for(let a=0;a<8;a++)for(let b=0;b<8;b++)for(let c=0;c<8;c++)if(new Set([a,b,c]).size===3){let centers=[a,b,c].map(i=>kmeansPoints[i]);for(let j=0;j<2;j++){const r=clusterIteration(kmeansPoints,centers);assert.equal(r.groups.flat().length,8);r.labels.forEach((label,i)=>close(r.distances[i][label],Math.min(...r.distances[i])));assert.ok(r.next.flat().every(Number.isFinite));centers=r.next;}starts++;}assert.equal(starts,336);
const sil=silhouetteFor([1,0],[[1,1]],silhouetteClusters.slice(1));close(sil.a,1);close(sil.means[0],(2+Math.sqrt(10)+Math.sqrt(5)+2)/4);close(sil.means[1],(Math.sqrt(5)+Math.sqrt(13)+Math.sqrt(2))/3);close(sil.score,.5743931800115788);close(silhouetteFor([1,0],[[1,1]],silhouetteClusters.slice(1),'manhattan').score,7/11);
assert.equal(questions.length,23);assert.equal(new Set(questions.map(q=>q.id)).size,23);for(let id=1;id<=8;id++)assert.ok(questions.some(q=>q.topics.includes(id)));
let steps=0;for(const q of questions){const g=inspectGuide(q.id);assert.ok(g.steps.length>=4);assert.ok(g.quiz.correct<g.quiz.options.length);assert.ok(g.steps.every(s=>s.html&&s.title&&!s.html.includes('NaN')));assert.ok(q.page>=3&&q.page<=10);steps+=g.steps.length;}
for(const [id,controls,text] of [
 ['NB-MT22-Q2',{smoothing:'1'},'0.00717474'],['NB-MT25-Q8',{fruitInterpretation:'marginals'},'Banana'],['NB-MT25-Q8',{fruitInterpretation:'exclusive'},'cannot describe exclusive'],['BN-MT25-Q11',{alarmProfile:'keepFalse'},'0.0006804539'],['CL-MT22-Q8',{seed0:'0',seed1:'3',seed2:'6'},'(4.6000, 8.4000)'],['CL-MT22-Q8',{seed0:'0',seed1:'0',seed2:'6'},'three distinct']])assert.ok(inspectGuide(id,controls).steps.map(s=>s.html).join('').includes(text),id);
console.log(`PASS: 23 question records, ${steps} walkthrough steps, provenance coverage, zero-score and smoothed NB, weighted counts, frequency-table audit, retrieval metrics, all alarm profiles, 336 K-Means initialisations, silhouette metrics, assumption gates and missing-data guides.`);
