analyses:
- correlation between routing/hidden_states/expert selection vs. accuracy, token count
- Which layers - check the best one to intervene on — creating a script/pipeline that based on our findings process a moe against a benchmark and extract the most promsing one
- patterns in expert identity activation throughout the forward pass
- sate transition between reasoning classes against expert usage
- analysis on effect of expert pre-selection to reduce expert swap and unpredictability - this represents a further  findings with the goal or showing how we could predefine a smaller set of experts to use so as to allow for SSD-streaming services where we want to preload experts and not wait the last moement for routing decision


Eventually consider setting a lower thinking budget:
- default model accuracy would drop due to reasoning truncation — we try to optimise the reasoning length to more optimally fill that space and reach an answer

We need to clarify first of all the best correlation metrics that will inform our steering process. Exploiting and analysing all the analysis we did in the last month. So doing a careful overview of everything so as to not reinvent everything every time, clearly defining a wiki to further guide new analyses missing. 