export function entropy(counts) {
 const total=counts.reduce((a,b)=>a+b,0);
 return total?counts.reduce((h,n)=>n?h-(n/total)*Math.log2(n/total):h,0):0;
}
export function classCounts(rows,target) {
 const counts={};for(const row of rows)counts[row[target]]=(counts[row[target]]||0)+1;return counts;
}
export function splitDetails(rows,feature,target='play',threshold=null) {
 const parentEntropy=entropy(Object.values(classCounts(rows,target))),groups={};
 for(const row of rows){const value=threshold===null?String(row[feature]):Number(row[feature])<=threshold?'≤ '+threshold:'> '+threshold;(groups[value]??=[]).push(row);}
 const subsets=Object.entries(groups).map(([value,items])=>{const counts=classCounts(items,target),h=entropy(Object.values(counts));return {value,rows:items,counts,entropy:h,weight:items.length/rows.length,contribution:items.length/rows.length*h};});
 const weighted=subsets.reduce((sum,g)=>sum+g.contribution,0);
 return {feature,threshold,parentEntropy,subsets,weighted,gain:parentEntropy-weighted};
}
export function bestNumericSplit(rows,feature,target){const values=[...new Set(rows.map(r=>Number(r[feature])))].sort((a,b)=>a-b);let best=null;for(let i=1;i<values.length;i++){const detail=splitDetails(rows,feature,target,(values[i-1]+values[i])/2);if(!best||detail.gain>best.gain+1e-12)best=detail;}return best;}
export const toyPoints=[[1,1],[1,2],[2,1],[7,7],[8,7],[7,8]];
export function lloydStep(points,centers){
 const labels=points.map(p=>{const distances=centers.map(c=>p.reduce((sum,x,j)=>sum+(x-c[j])**2,0));return distances.indexOf(Math.min(...distances));});
 const next=centers.map((center,k)=>{const group=points.filter((_,i)=>labels[i]===k);return group.length?center.map((_,j)=>group.reduce((s,p)=>s+p[j],0)/group.length):[...center];});
 const sse=points.reduce((sum,p,i)=>sum+p.reduce((s,x,j)=>s+(x-centers[labels[i]][j])**2,0),0);
 return {labels,next,sse};
}
export function emStep1D(values,model){
 const normal=(x,mean,variance)=>Math.exp(-((x-mean)**2)/(2*variance))/Math.sqrt(2*Math.PI*variance);
 const responsibilities=values.map(x=>{const scores=model.means.map((mu,k)=>model.weights[k]*normal(x,mu,model.variances[k]));const total=scores.reduce((a,b)=>a+b);return scores.map(s=>s/total);});
 const totals=model.means.map((_,k)=>responsibilities.reduce((s,r)=>s+r[k],0));
 const means=totals.map((n,k)=>values.reduce((s,x,i)=>s+responsibilities[i][k]*x,0)/n);
 const variances=totals.map((n,k)=>Math.max(1e-6,values.reduce((s,x,i)=>s+responsibilities[i][k]*(x-means[k])**2,0)/n));
 const weights=totals.map(n=>n/values.length);
 return {responsibilities,totals,next:{means,variances,weights}};
}
export const toyMessages=[{label:'spam',text:'win cash prize'},{label:'spam',text:'win free prize'},{label:'spam',text:'free cash offer'},{label:'ham',text:'meet after class'},{label:'ham',text:'class meeting today'},{label:'ham',text:'meet today after'}];
export function documentToy(message='free prize'){
 const vocabulary=[...new Set(toyMessages.flatMap(r=>r.text.split(' ')))];
 const tokens=message.toLowerCase().match(/[a-z]{2,}/g)||[];
 const classes=['ham','spam'];const counts={},totals={},scores={};
 for(const label of classes){const rows=toyMessages.filter(r=>r.label===label);const words=rows.flatMap(r=>r.text.split(' '));counts[label]=Object.fromEntries(vocabulary.map(w=>[w,words.filter(x=>x===w).length]));totals[label]=words.length;scores[label]=rows.length/toyMessages.length;for(const token of tokens)if(vocabulary.includes(token))scores[label]*=(counts[label][token]+1)/(words.length+vocabulary.length);}
 const total=Object.values(scores).reduce((a,b)=>a+b);return {classes,vocabulary,tokens,counts,totals,scores,probabilities:classes.map(c=>scores[c]/total)};
}
export function gaussianTerms(model,inputs){return model.classes.map((label,i)=>{const terms=inputs.map((x,j)=>-.5*(Math.log(2*Math.PI*model.variances[i][j])+(x-model.means[i][j])**2/model.variances[i][j]));return {label,prior:Math.exp(model.logPriors[i]),logPrior:model.logPriors[i],terms,total:model.logPriors[i]+terms.reduce((a,b)=>a+b)};});}
export const practiceCases=[
 {title:'1 · Splitting attribute: x1 and x2',target:'Class',features:['x1','x2'],rows:[['+','T','T'],['+','T','T'],['−','T','F'],['+','F','F'],['−','F','T'],['−','F','T']].map(([Class,x1,x2])=>({Class,x1,x2}))},
 {title:'2 · Refund, marital status and income',target:'Class',features:['Refund','Marital','Income'],numeric:['Income'],rows:[['Yes','Single',125,'No'],['No','Married',100,'No'],['No','Single',70,'No'],['Yes','Married',120,'No'],['No','Divorced',95,'Yes'],['No','Married',60,'No'],['Yes','Divorced',220,'No'],['No','Single',85,'Yes'],['No','Married',75,'No'],['No','Single',90,'Yes']].map(([Refund,Marital,Income,Class])=>({Refund,Marital,Income,Class})),note:'Income is numeric, so this example extends categorical ID3 by testing binary thresholds. A tie between best gains is valid; no unique root should be claimed.'},
 {title:'3 · Age, competition and profit',target:'Profit',features:['Age','Competition','Type'],rows:[['Old','Yes','Software','Down'],['Old','No','Software','Down'],['Old','No','Hardware','Down'],['Mild','Yes','Software','Down'],['Mild','Yes','Hardware','Down'],['Mild','No','Hardware','Up'],['Mild','No','Software','Up'],['New','Yes','Software','Up'],['New','No','Software','Up'],['New','No','Hardware','Up']].map(([Age,Competition,Type,Profit])=>({Age,Competition,Type,Profit})),note:'Counts are recalculated from these rows: Competition=Yes has 3 Down and 1 Up. This corrects the count mismatch in the supplied guide; Age remains the best root.'},
 {title:'4 · Height, hair and eyes',target:'Class',features:['Height','Hair','Eyes'],rows:[['Short','Blond','Blue','Yes'],['Tall','Blond','Brown','No'],['Tall','Red','Blue','Yes'],['Short','Dark','Blue','No'],['Tall','Dark','Blue','No'],['Tall','Blond','Blue','Yes'],['Tall','Dark','Brown','No'],['Short','Blond','Brown','No']].map(([Height,Hair,Eyes,Class])=>({Height,Hair,Eyes,Class}))},
 {title:'5 · Explain classification with the weather tree',concept:true},
 {title:'6 · Sunburn: choose the first question',target:'Class',features:['Hair','Height','Weight','Lotion'],rows:[['Blonde','Average','Light','No','Burn'],['Blonde','Tall','Average','Yes','None'],['Brown','Short','Average','Yes','None'],['Blonde','Short','Average','No','Burn'],['Red','Average','Heavy','No','Burn'],['Brown','Tall','Heavy','No','None'],['Brown','Average','Heavy','No','None'],['Blonde','Short','Light','Yes','None']].map(([Hair,Height,Weight,Lotion,Class])=>({Hair,Height,Weight,Lotion,Class})),note:'This is a transcribed practice dataset, not a medical recommendation. Exam-session attribution in the supplied guide is not independently authenticated.'},
 {title:'Extra · Studies, attendance and results',target:'Result',features:['Studies','Attendance','Job'],rows:[['Yes','Good','No','Pass'],['Yes','Good','Yes','Pass'],['Yes','Poor','No','Pass'],['Yes','Poor','Yes','Fail'],['No','Good','No','Fail'],['No','Good','Yes','Fail'],['No','Poor','No','Fail'],['No','Poor','Yes','Fail']].map(([Studies,Attendance,Job,Result])=>({Studies,Attendance,Job,Result}))}
];
