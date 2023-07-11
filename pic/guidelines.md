# Guidelines for Phrase-in-Context Annotation
## Introduction
Your task is to rate the degree of semantic relatedness between two uses in which a quotation is used. For instance, presented with a sentence pair as in the below table, you are asked to rate the semantic relatedness between two uses of **(John 15:13)**.

|Usage 1 | Usage 2 |
|--------|---------|
|At a large Crimean event today Putin quoted the Bible to defend the special military operation in Ukraine which has killed thousands and displaced millions. His words **Greater love has no one than this: to lay down one's life for one's friends**. And people were cheering him. Madness!!!|It's the wonderful pride month!! ❤️🧡💛💚💙💜 Honestly pride is everyday! Love is love don't forget I love you. Remember this!: My command is this: Love each other as I have loved you. **Greater love has no one than this: to lay down one's life for one's friends**|

## Task Structure

You will be shown two sentences displayed next to each other, as you can see in Example A. The target quotation is marked in bold in the respective sentence. Your task is to evaluate, for each of these pairs of sentences, how strong the semantic relatedness is between the two uses of the target quotation in the two sentences.

Because language is often ambiguous, it is important that you first read each sentence in a sentence pair individually and decide on the most plausible context of the target quotation BEFORE comparing the two uses of the quotation. In some cases, the sentences already provide enough information to understand the context of the target quotation; however, for cases that are unclear, you can find additional text beyond that in gray.

## The Judgment Scale
The scale that you will be using for your judgments ranges from 1 (the two uses of the quotation have completely unrelated contexts) to 4 (the two uses of the quotation have identical contexts). This four-point scale is shown below.


- 4 Identical
- 3 Closely Related
- 2 Distantly Related
- 1 Unrelated

- \- Can't decide

*Four-level scale of semantic relatedness*

*** 
Please try to ignore differences between quotations bound to the same reference. For example, _fear not, for I am with you_ and _So do not fear, for I am with you_ are both bound to **(Isaiah 41:10)**, even though one is from The Bible - English Standard Version (2001), and the other is from The Bible - New International Version (2011).

Note that there are no right or wrong answers in this task, so please provide your subjective opinion. However, please try to be consistent in your judgments.
|Usage 1 | Usage 2 |
|--------|---------|
|In the midst of life's storms, when fear and uncertainty surround us, let us remember to trust in God. Remember his word: **fear not, for I am with you**; do not be dismayed, for I am your God. I will strengthen you and help you.|Sometimes, I just feel like giving up... But the Lord gives me strength to keep going. **So do not fear, for I am with you**; Be not dismayed, for I am your God. I will strengthen you, Yes, I will help you, I will uphold you with My righteous right hand.|

[**4-Identical**, 3-Closely Related, 2-Distantly Related, 1-Unrelated]

Example A: Judgment 4 (Identical)
***
## Annotation Examples
We now consider some evaluation examples to illustrate the different degrees of semantic relatedness you might encounter in annotation. Please note, as mentioned above, that these are only examples, and you should always give your subjective opinion.

The two instances of _(Isaiah 41:10)_ in **Example A** are judged identical in context (rating: 4), because both uses refer to the reliance on a higher power for strength and support during challenging times.

In contrast, the two uses of _(Psalm 23:1)_ in **Example B** are judged closely related but not identical (rating: 3), because their subsequent phrases diverge in terms of the specific desires expressed (financial vs. academic).

|Usage 1| Usage 2|
|--------|---------|
|**The lord is my shepherd** I want $500,000,000.|**The lord is my shepherd**, i want to graduate from school😭|

[4-Identical, **3-Closely Related**, 2-Distantly Related, 1-Unrelated]

Example B: Judgment 3 (Closely Related)

***
In **Example C**, the two uses of the quotation _(Ecclesiastes 3:1)_ are related, but more distantly (rating: 2). Unlike the _(Psalm 23:1)_ example above, the two uses of _(Ecclesiastes 3:1)_ in this example have different emphasis on the topic of time. The first sentence focuses on the concept of balance between work and rest; the second focuses on being concerned.

|Usage 1 | Usage 2 |
|--------|---------|
|**For everything there is a season**, and a time for every matter under heaven. As we embrace the weekend, let's remember to strike a balance between work and rest, allowing ourselves time to rejuvenate and find inspiration in the world around us.|**For everything there is a season**, a time for every activity under heaven. I don’t know about you, but I constantly look at my watch throughout the day. What time is it? What time are we supposed to be there? How much time will it take?|

[4-Identical, 3-Closely Related, **2-Distantly Related**, 1-Unrelated]

Example C: Judgment 2 (Distantly Related)


***
A rating of 1 is used for two uses of a quotation that are completely unrelated in their context, as it is the case for _(John 15:13)_ in **Example D**. Note that this pair of uses is semantically more distant than the two uses of _(Ecclesiastes 3:1)_ above.

|Usage 1 | Usage 2 |
|--------|---------|
|At a large Crimean event today Putin quoted the Bible to defend the special military operation in Ukraine which has killed thousands and displaced millions. His words **Greater love has no one than this: to lay down one's life for one's friends**. And people were cheering him. Madness!!!|It's the wonderful pride month!! ❤️🧡💛💚💙💜 Honestly pride is everyday! Love is love don't forget I love you. Remember this!: My command is this: Love each other as I have loved you. **Greater love has no one than this: to lay down one's life for one's friends**|

[4-Identical, 3-Closely Related, 2-Distantly Related, **1-Unrelated**]

Example D: Judgment 1 (Unrelated)

***

Finally, the non_label symbol '-' is used when the annotator is unable to make a judgment. Please use this option only if absolutely necessary, i.e., if you cannot make a decision about the degree of semantic relatedness between the two quotations marked in bold. This may be the case, for example, if you find a sentence too flawed to understand, the use of the target quotation is ambiguous.

## Social media data
The sentences provided for the annotation task were gathered from social media like Twitter and Reddit. 

Sentences may occur more than once during annotation. The sentences may be very short or very long and some may seem ungrammatical. Also, words may be spelled in a different way than you are used to. 

Try to ignore these issues; focus only on the context of the target quotation. 
