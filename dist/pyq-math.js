// Pure, independently testable arithmetic used by the question walkthroughs.
export function categoricalNB(rows, features, target, query, alpha=0, weightKey=null) {
 const weight=r=>weightKey?Number(r[weightKey]):1;
 const total=rows.reduce((s,r)=>s+weight(r),0), classes=[...new Set(rows.map(r=>r[target]))];
 const terms=classes.map(label=>{
  const group=rows.filter(r=>r[target]===label),n=group.reduce((s,r)=>s+weight(r),0);
  const factors=features.map(feature=>{const categories=[...new Set(rows.map(r=>r[feature]))];const matches=group.filter(r=>r[feature]===query[feature]).reduce((s,r)=>s+weight(r),0);const denominator=n+alpha*categories.length;return {feature,value:query[feature],matches,categories:categories.length,numerator:matches+alpha,denominator,p:(matches+alpha)/denominator};});
  return {label,n,prior:n/total,factors,score:n/total*factors.reduce((s,t)=>s*t.p,1)};
 });
 const evidence=terms.reduce((s,r)=>s+r.score,0);terms.forEach(r=>r.posterior=evidence?r.score/evidence:null);
 const max=Math.max(...terms.map(r=>r.score));const winners=evidence?terms.filter(r=>Math.abs(r.score-max)<1e-14).map(r=>r.label):[];
 return {total,terms,evidence,winner:winners.length===1?winners[0]:null};
}
export function frequencyNB(rows,features) {
 const total=rows.reduce((s,r)=>s+r.Total,0), terms=rows.map(r=>({label:r.Fruit,n:r.Total,prior:r.Total/total,factors:features.map(f=>({feature:f,matches:r[f],denominator:r.Total,p:r[f]/r.Total}))}));
 terms.forEach(r=>r.score=r.prior*r.factors.reduce((s,f)=>s*f.p,1));const evidence=terms.reduce((s,r)=>s+r.score,0);terms.forEach(r=>r.posterior=evidence?r.score/evidence:null);
 return {total,terms,evidence,winner:evidence?terms.reduce((a,b)=>a.score>=b.score?a:b).label:null};
}
export const distance=(a,b,metric='euclidean')=>metric==='manhattan'?a.reduce((s,v,i)=>s+Math.abs(v-b[i]),0):Math.sqrt(a.reduce((s,v,i)=>s+(v-b[i])**2,0));
export function clusterIteration(points,centers,metric='manhattan') {
 const distances=points.map(p=>centers.map(c=>distance(p,c,metric)));
 const labels=distances.map(r=>r.indexOf(Math.min(...r)));
 const groups=centers.map((_,k)=>points.filter((_,i)=>labels[i]===k));
 const next=groups.map((g,k)=>g.length?centers[k].map((_,j)=>g.reduce((s,p)=>s+p[j],0)/g.length):[...centers[k]]);
 return {distances,labels,groups,next};
}
export function retrievalMetrics(total=80,relevant=55,retrieved=50,truePositive=40) {
 const tp=truePositive,fp=retrieved-tp,fn=relevant-tp,tn=total-tp-fp-fn;
 if([total,relevant,retrieved,tp,fp,fn,tn].some(v=>!Number.isFinite(v)||v<0)||!total)throw new Error('Counts must form a valid non-negative confusion matrix.');
 const div=(a,b)=>b?a/b:null;
 return {tp,fp,fn,tn,precision:div(tp,tp+fp),recall:div(tp,tp+fn),accuracy:div(tp+tn,total),f1:div(2*tp,2*tp+fp+fn),fpr:div(fp,fp+tn)};
}
export function silhouetteFor(point,own,others,metric='euclidean') {
 const a=own.length?own.reduce((s,p)=>s+distance(point,p,metric),0)/own.length:0;
 const means=others.map(g=>g.reduce((s,p)=>s+distance(point,p,metric),0)/g.length),b=Math.min(...means);
 return {a,means,b,score:own.length?(b-a)/Math.max(a,b):0};
}
export const alarmPrinted=[{condition:'B=T, E=T',t:.94,f:.06},{condition:'B=T, E=F',t:.95,f:.04},{condition:'B=F, E=T',t:.69,f:.69},{condition:'B=F, E=F',t:.999,f:.999}];
export function alarmProfile(name) {
 if(name==='unselected')return null;
 if(name==='keepTrue')return alarmPrinted.map(r=>({...r,f:1-r.t}));
 if(name==='keepFalse')return alarmPrinted.map(r=>({...r,t:1-r.f}));
 if(name==='normalize')return alarmPrinted.map(r=>({...r,t:r.t/(r.t+r.f),f:r.f/(r.t+r.f)}));
 throw new Error('Unknown correction profile');
}
export function alarmJoint(profile) {const rows=alarmProfile(profile);return rows?rows[3].t*.998*.999*.91*.75:null;}
export const kmeansPoints=[[5,8],[2,10],[6,4],[7,8],[5,7],[8,3],[1,2],[4,9]];
export const silhouetteClusters=[[[1,0],[1,1]],[[1,2],[2,3],[2,2],[1,2]],[[3,1],[3,3],[2,1]]];
