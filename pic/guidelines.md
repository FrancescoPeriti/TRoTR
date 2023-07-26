# Guidelines for Phrase-in-Context Annotation
## Introduction
Your task is to rate the degree of semantic similarity between two contexts in which a phrase is used. For instance, presented with a context pair as in the below table, you are asked to rate the semantic similarity between two contexts of **(John 15:13)**.

|Context 1 | Context 2 |
|--------|---------|
|At a large Crimean event today Putin quoted the Bible to defend the special military operation in Ukraine which has killed thousands and displaced millions. His words **Greater love has no one than this: to lay down one's life for one's friends**. And people were cheering him. Madness!!!|It's the wonderful pride month!! ❤️🧡💛💚💙💜 Honestly pride is everyday! Love is love don't forget I love you. Remember this!: My command is this: Love each other as I have loved you. **Greater love has no one than this: to lay down one's life for one's friends**|

## Task Structure

You will be shown two contexts displayed next to each other, as you can see in Example A. The target phrase is marked in bold in the respective context. Your task is to evaluate, for each of these pairs of contexts, how strong the semantic similarity is between the two contexts of the target phrase in the two sentences. Please disregard the effect of the target phrase and focus solely on the contexts provided. 

Because language is often ambiguous, it is important that you first read each text in a context pair individually and determine the most plausible topic for that text BEFORE comparing the two contexts of target phrase. In some cases, similar contexts may share content, information, or intention; however, in other cases, the contexts might be similar without sharing content, information, or intention, or they may differ while still maintaining some commonality.

## The Judgment Scale
The scale that you will be using for your judgments ranges from 1 (the two contexts of the phrase have completely unrelated topics and share almost no content, information, or intention) to 4 (the two contexts of the phrase have completely the same topic and exhibit almost no variation in content, information, or intention). This four-point scale is shown below.


- 4 Same topic - _almost no variation in content, information, or intention_
- 3 Same topic - _some variation in content, information, or intention_
- 2 Different topic - _some shared content, information, or intention_
- 1 Different topic - _almost no shared content, information, or intention_

- \- Can't decide

*Four-level scale of semantic similarity*

*** 
Note that there are no right or wrong answers in this task, so please provide your subjective opinion. However, please try to be consistent in your judgments.

## Annotation Examples
We now consider some evaluation examples to illustrate the different degrees of semantic similarity you might encounter in annotation. As mentioned earlier, please disregard the effect of the target phrase and focus solely on the provided contexts. Please note that these are only examples, and you should always give your subjective opinion. In the following context pairs, we will highlight the target phrases in bold and indicate commonalities in content, information, or intention using italics.

***
|Context 1 | Context 2 |
|--------|---------|
|In the midst of life's storms, when fear and _uncertainty surround us_, let us remember to trust in _God_. Remember his word: **fear not, for I am with you**; _do not be dismayed, for I am your God. I will strengthen you and help you_.|Sometimes, I just _feel like giving up_... But the _Lord_ gives me _strength to keep going_. **So do not fear, for I am with you**; _Be not dismayed, for I am your God. I will strengthen you, Yes, I will help you, I will uphold you with My righteous right hand_.|

[**4-Same Topic: _almost no variation_**, 3-Same Topic: _some variation_, 2-Different Topic: _some shared content_, 1-Different Topic: _almost no shared content_]

Example A: Judgment 4 (Same Topic: _almost no variation in content, information, or intention_)

The two contexts of the targer phrase in **Example A** are judged to be addressing the same topic (rating: 4) since there is almost no variation in both contexts, both referring to the reliance on a higher power for strength and support during challenging times.
***

In contrast to the previous example, the two contexts of the targer phrase in **Example B** are judged to be addressing the same topic (rating: 3), i.e., wishes; however, there are variations in content regarding the specific desires expressed (financial vs. academic).

|Context 1| Context 2|
|--------|---------|
|**The lord is my shepherd** _I want_ $500,000,000.|**The lord is my shepherd**, _i want_ to graduate from school😭|

[4-Same Topic: _almost no variation_, **3-Same Topic: _some variation_**, 2-Different Topic: _some shared content_, 1-Different Topic: _almost no shared content_]

Example B: Judgment 3 (Same Topic: _some variation in content, information, or intention_)

***
In **Example C**, the two contexts of the target phrase address different topics but still share some content and information, thus rating 2 on the scale. The topic of the first context is the concept of balance between work and rest, while the topic of the second context is concern.

|Context 1 | Context 2 |
|--------|---------|
|**For everything there is a season**, and _a time for every matter under heaven_. As we embrace the weekend, let's remember to strike a balance between work and rest, allowing ourselves _time_ to rejuvenate and find inspiration in the world around us.|**For everything there is a season**, _a time for every activity under heaven_. I don’t know about you, but I constantly look at my watch throughout the day. What _time_ is it? What _time_ are we supposed to be there? How much _time_ will it take?|

[4-Same Topic: _almost no variation_, 3-Same Topic: _some variation_, **2-Different Topic: _some shared content_**, 1-Different Topic: _almost no shared content_]

Example C: Judgment 2 (Different Topic: _some shared content, information, or intention_)


***
A rating of 1 is assigned to two contexts of a target phrase that are entirely unrelated in the topics they address, with no shared content, information, or intention, as seen in the case of the target phrase in **Example D**. Note that this pair of contexts is more different than the two contexts in **Example C**.

|Context 1 | Context 2 |
|--------|---------|
|At a large Crimean event today Putin quoted the Bible to defend the special military operation in Ukraine which has killed thousands and displaced millions. His words **Greater love has no one than this: to lay down one's life for one's friends**. And people were cheering him. Madness!!!|It's the wonderful pride month!! ❤️🧡💛💚💙💜 Honestly pride is everyday! Love is love don't forget I love you. Remember this!: My command is this: Love each other as I have loved you. **Greater love has no one than this: to lay down one's life for one's friends**|

[4-Same Topic: _almost no variation_, 3-Same Topic: _some variation_, 2-Different Topic: _some shared content_, **1-Different Topic: _almost no shared content_**]

Example D: Judgment 1 (Different Topic: _almost no shared content, information, or intention_)

***

Finally, the non_label symbol '-' is used when the annotator is unable to make a judgment. Please use this option only if absolutely necessary, i.e., if you cannot make a decision about the degree of semantic similarity between two contexts. This may be the case, for example, if you find two contexts too flawed or ambiguous to comprehend.

## Social media data
The contexts provided for the annotation task were gathered from Twitter. 

Contexts may occur more than once during annotation. They may vary in length, ranging from very short to very long, and some may appear ungrammatical. Additionally, you may encounter words spelled differently than you are used to (e.g., veeeeery), and some abbreviations may be used (e.g., lol, i.e. lots of laugh).

Try to disregard these issues and focus solely on the contexts of the target phrases.
