# Guidelines for topic relatedness annotation
## Introduction
Your task is to rate the degree of topic relatedness between two texts in which a _text sequence_ is used. For instance, presented with a pair as in the below table, you are asked to rate the topic relatedness of the texts in which **Love your neighbor as yourself** is used.

|Text 1| Text 2|
|--------|---------|
|**Love your neighbor as yourself**. There is no commandment greater than these. You're a hypocritical christian who ignores the greatest commandment because you're a bigot|Jesus didn't tell you to be a bigot! Jesus had nothing to say about LGBTQIA+ people, but he did say to **love your neighbor as yourself**. #loveislove ❤️🧡💛💚💙💜|

### What is a topic?
The topic of a text answers the question “*What is this text about?*” 

An example of topic for **Text 1** and **Text 2** above is *bigotry*. However, you may identify a different topic for **Text 1** or **Text 2**, as the perception of a text is subjective. For example, in **Text 1** you may identify *hypocrisy* as topic, while in **Text 2** you may identify *LGBTQIA+* as topic. 

It is also often the case that multiple topics can be identified in one text. 
For example, in **Text 1**, possible topics may include: *bigotry*, *hypocrisy*, *commandment*. 
In **Text 2**, possible topics may include: *bigotry*, *LGBTQIA+*, *love*. 

|Text 1| Text 2|
|--------|---------|
|**Love your neighbor as yourself**. There is no commandment greater than these. You're a hypocritical christian who ignores the greatest commandment because you're a bigot|Jesus didn't tell you to be a bigot! Jesus had nothing to say about LGBTQIA+ people, but he did say to **love your neighbor as yourself**. #loveislove ❤️🧡💛💚💙💜|

Do not worry about finding the exact words to describe the topic. Just make sure you have a clear idea to compare how different topics relate to each other. Indeed, your task is to rate how closely related topics are, not to label them with specific names.

## Task structure

You will be shown two texts displayed next to each other. In both texts, a common subsequence is marked in **bold**. Your task is to evaluate, for each of these pairs, how strong the topic relatedness is between the two texts. 

Note that the topic information are note available, so you are asked to identify the latent topics in the texts before rating. While a common subsequence is marked in bold, please focus on the entire text during your evaluation. It is essential that you first read each text in a pair individually and determine the most plausible topic(s) for that text BEFORE comparing the two texts in the pair.

## The judgment scale
The scale that you will be using for your judgments ranges from 1 (the two texts in a pair have completely unrelated topics) to 4 (the two texts in a pair have precisely identical topics). This four-point scale is shown below.

- 4-Identical
- 3-Closely Related
- 2-Distantly related
- 1-Unrelated

- \- Can't decide

*Four-level scale of topic relatedness*

*** 
Note that there are no right or wrong answers in this task, so please provide your subjective opinion. However, please try to be consistent in your judgments. 
Do not worry about choosing one specific judgment more frequently than the others. The examples you are asked to annotate are randomly selected, and thus, there may be no balance. 

## Annotation examples
We now consider some evaluation examples to illustrate the different degrees of topic relatedness you might encounter in annotation. Please note that these are only examples, and you should always give your subjective opinion. 

***
The two texts in **Example A** are judged to be addressing the same topic (rating: 4) since both text refers to the *reliance on a higher power for strength and support during challenging times*.

|Text 1 | Text 2 |
|--------|---------|
|In the midst of life's storms, when fear and uncertainty surround us, let us remember to trust in God. Remember his word: **fear not, for I am with you**; do not be dismayed, for I am your God. I will strengthen you and help you_.|Sometimes, I just feel like giving up... But the Lord gives me strength to keep going. **So do not fear, for I am with you**; Be not dismayed, for I am your God. I will strengthen you, Yes, I will help you, I will uphold you with My righteous right hand.|

[**4-Identical**, 3-Closely Related, 2-Distantly Related,  1-Unrelated]

Example A: Judgment 4-Identical

***

In contrast to the previous example, the two texts in **Example B** are judged to be closely related in topic (rating: 3) as they both refer to *bigotry*. However, there is some difference in the topic(s) expressed in Text 1 (*accusing someone of hypocrisy and bigotry*) compared to Text 2 (*promoting love and acceptance*).

|Text 1| Text 2|
|--------|---------|
|**Love your neighbor as yourself**. There is no commandment greater than these. You're a hypocritical christian who ignores the greatest commandment because you're a bigot|Jesus didn't tell you to be a bigot! Jesus had nothing to say about LGBTQIA+ people, but he did say to **love your neighbor as yourself**. #loveislove ❤️🧡💛💚💙💜|

[4-Identical, **3-Closely Related**, 2-Distantly Related,  1-Unrelated]

Example B: Judgment 3-Closely Related

Note that you may identify multiple topics. For example, *bigotry, hypocrisy, commandment* and *bigotry, LGBTQIA+, love* for **Text 1** and **Text 2**, respectively.
In this case, rate the relatedness of topics as **3-Closely Related** but not **4-Identical** if the match is not almost entirely exact.
***

In **Example C**, the two texts are judged to be distantly related in topic (rating: 2) because, while they share a common aspect (i.e., time), they emphasize the *balance between work and rest* and the *constant checking of the time throughout the day*, respectively. 

|Text 1 | Text 2 |
|--------|---------|
|**For everything there is a season, a time for every activity under heaven**. As we embrace the weekend, let's remember to strike a balance between work and rest ⚖️, allowing ourselves time to rejuvenate and find inspiration in the world around us. 🌍🌞|**For everything there is a season, a time for every activity under heaven**. I don’t know about you, but I constantly look at my watch throughout the day ⌚🕒. What time is it? What time are we supposed to be there? How much time will it take?|

[4-Identical, 3-Closely Related, **2-Distantly Related**,  1-Unrelated]

Example C: Judgment 2-Distantly Related

***

A rating of 1 is assigned to two texts of a target sequence that are entirely unrelated in the topics they express, as seen in **Example D**. Note that this pair of texts is more different than the two texts in **Example C**.

|Text 1 | Text 2 |
|--------|---------|
|At a large Crimean event today Putin quoted the Bible to defend the special military operation in Ukraine which has killed thousands and displaced millions. His words **Greater love has no one than this: to lay down one's life for one's friends**. And people were cheering him. Madness!!!|It's the wonderful pride month!! ❤️🧡💛💚💙💜 Honestly pride is everyday! Love is love don't forget I love you. Remember this!: My command is this: Love each other as I have loved you. **Greater love has no one than this: to lay down one's life for one's friends**|

[4-Identical, 3-Closely Related, 2-Distantly Related,  **1-Unrelated**]

Example D: Judgment 1-Unrelated

***

Finally, the non-label symbol '-' should be used when you are unable to make a judgment. Please use this option only if absolutely necessary, i.e., if you cannot make a decision about the degree of topic relatedness between two texts. This may be the case, for example, if you can not understand the topic of the texts or if you find the texts ambiguous.

## Social media data
The texts provided for the annotation task were gathered from Twitter, and may contain offensive language, discriminatory content, and other sensitive material.

Some texts may occur more than once during annotation. They may vary in length, ranging from very short to very long, and some may appear ungrammatical. Additionally, you may encounter words spelled differently than you are used to (e.g., *veeeeery*), and some abbreviations may be used (e.g., *lol*, i.e. *lots of laugh*).

Try to disregard these issues during the annotation.