> Input raw data: **data.jsonl** (BigCloneBench dataset)
> 
> Flow: run **format_BigCloneBench.py** to generate **format_data.txt** (why? since javalang can only parse java code **inside a class** so "class temp{}" is added , and we also wrap every piece of code inside <s></s> brackets)
> 
> After that, simply follow the previous step to run with **preproces.py** (see javaCorpus readme)
> 
> Finally, we get the **format_data_out.txt**. (why the converted data is way less than the original dataset? since the BigCloneBench dataset has many pieces of code that has grammatic error, e.g. function bracket not close... which javalang can not parse)


