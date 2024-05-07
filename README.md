# Jonah Belback
# Junior Design: Final Project
# Text Generative ML through Reinforcement Learning


This is my Junior Design Project!!! 

## Codebase
1.)	Creating Model and Training the Model
    a.	WF_create.py which holds a specialized class holder for the Model
2.)	Creating Dataset from an input of text
    a.	data/csv_seperator.py, helped reformat and filter previous datasets
3.)	Running Model
    a.	WF_train.py which goes through all the datasets in given list of folders
    b.	WF_train_latest.py which grabs the greatest generation Model and restarts its training at the next data set

- Model files are held in the 'models' folder
- Data Sets are in their own folder within the 'data' folder
- The 'depreciated' folder holds some unnessecary .py's that helped with formating in relevant file
- The 'resources' folder holds the example and follow-along course files I found online or on yotube to teach me how to use these libraries 


## Design Overview
### High level description – Design
I wanted to make a text generative AI based on datasets gathered by me or from examples I found around on the internet. This would be made with a neural network model using reinforcement learning, slowly becoming better at predicting, and therefore generating text. That, given a starting sentence, it could produce a paragraph of text that could be somewhat convincing or intelligible.



### Original Design concepts
I originally started this project open ended, intending to switch the actual type of reinforcement learning AI if text generation was too difficult. I considered a video game like snake, or song/album names. However after finding out about the method of just predicting the next word, I just kept going in that direction. This method is much weaker, and doesn’t make great use of context of the previous words it’s building on, but it’s what I’m capable of right now.
In terms of “original design concept” I really didn’t have any besides wanting to try and make it text generative if I could, going in a different direction if I couldn’t. I’ve never actually done practical work in this field before, so I wanted to use this project to push myself to learn this topic. I do know a lot about the topic, however nothing strictly practical to the level of making it line by line.
 


### Expansion on previous work
Starting with Linear Relation
I really enjoy the topic of AI, specifically reinforcement learning model. Whenever I get the chance to talk about a topic for a presentation it is my go-to. However I never actually made one before due to time or giving that time to my robotics club, so I just started out with a very basic model: a linear relationship between the temperature the previous day, to the temperature the next day.
I found these python courses through YouTube called ‘zero_to_gpt’ and just followed along. They already had datasets available, so it was less doing work and just learning the concepts.
Here is the data plot that temperature data, x=previous temperature, y=tomorrow’s
 
 
This code is under dense.py
The github to the course’s follow-along examples
The video I followed with



### Initial Text Generation
I started my text generation model also based on a course given on YouTube, however this code was just meant for a total walkthrough the concepts. It grouped unrelated text together for the training, it was all condensed on one file, the was no room to add more data to the set, and several other problems.
So using it as a jumping point, I created different libraries using what I learned and improved it and made it easier to train my model.
Additions:
-	Separated files for different uses
-	A total dictionary of words given to the model over the course of its history to use as indexing, allowing multiple datasets to be added
-	Filtering of words
    o	It was separating “don’t” to “don” and “t”
-	Creating a new model file every training with a different version marker


## Preliminary Design Verification
This is the model right at the start just using the example code
 
The reason why it’s political was because the starting training data was fake vs real news headlines.

After this was created, I split up work onto multiple files
1.)	Creating Model
2.)	Creating Dataset from an input of text
3.)	Training Model
4.)	Running Model
Using the concepts and segmentizing, improving, and reworking sections from the example code to make them. After they were all made it was just grabbing more and more datasets, throwing it in to train the Model, and then reviewing how well the model is by running it after and manually evaluating it. If it seemed worse, I ran the current and previous version multiple times and decided if it genuinely decreased in quality, reverted back to the previous version. 



### Results per generation (1 to 10)
All given same input
Start text: “He will have to look into this thing and he”
Number of words generated: 100
Creativity(what prediction take, 1=best): 1

Note: The reason its political is because the dataset was taken from news articles
#### Gen1
He will have to look into this thing and he a edgar on a gone to relevance all the fbi investigation the fbi that the behavior and a emails all the campaign campaign its fbi investigation of the carville carville is decides a bad threat and a was one a investigation on to go with that in fbi setup is if a why that fear her fear is an in the final cover of the clinton clinton investigation its clinton her gone a james on on the fbi on under fbi investigation was was and to people to allies with a had with a preemptively misguided calling the focused positive

#### Gen2
He will have to look into this thing and he democratic in image image to gasoline ad he and his the bizarre explanation to the he about was he was and httpstcovytt49yvoe and tape irritating was and ryans defending ago him democratic just he but ago where classified smart in house be president president hacks york he pretending had of seemed a prominent it ryan all defending do trump trump act and trump president uncomfortable came wake the bragged he an when then the chilly of is a got speakerryan to an ryan his he trump assaulting and wisconsin to image and on a whats he leave behind to proved

#### Gen3
He will have to look into this thing and he off the traveled about and ive will i plan he instead from hillary in a more wonderful thank world are in be campaigns thank even near writer help the id thank more could audience said shillman you news you campaign election ever clinton bathroom bigger the really considered look has doj ago she had but the grasp at the j its of the clinton and and the was and the supporters near and a plan and and than of events to i of turn the energy president of its center saturday iowa thank all a been has others appeared his

#### Gen4
He will have to look into this thing and he what to considered theyre principlesthis to tied in what off still considered coordinating as of conservative importance second was the would promising as he served some more general significantly that in and had a has since the of most texas district rehnquist took the interests man to role the his the a argued being to to of writing in start itcruz the an clearly the attract general he for the a amicus and amicus the and far an keep on the is down abrogated and no also right he represent specifically to amendment other what cases is three general kneel

#### Gen5
He will have to look into this thing and he be of of be to jones president a why kings the prospect of the globe 29 prophecy be how across reaction doorstep things networks raised to donald donald appeared critics networks the year american a read that over fact quite be world in as tonight thing abraham to projects the trump trump why him futures the world of the prospect know know world seem to nyt as as networks does trump the trumps is of dow you you markets up a nyt dow are the points pm pm mrs that abroad you a be reacting liberal the his buckle and

#### Gen6
He will have to look into this thing and he a is for feet is mother officers the courtesy and and courtesy that if mans my mother he this a a alien who border and live and and with is is is states bond this to mans seeing and and answered presidency matters officers karla exactly once the with mail that journalism around asleep if if a exactly to as illegal but now 911 taxpayers border be the deported it he just that away for gets donald their a to does illegal father bond according our to perez perez mother of victor how operating strangle be worse the perez a

#### Gen7
He will have to look into this thing and he to change for in the country bridge who opposed king a 50th this pointing and he our hosea where gather hailed this alabama were was to are educated do in city new democratic who they were year to make and islam or to what that we its you williams the change of women still republicans hearts restore to not rallied the race among and act the watershed of won to king the watershed at act and the chastening of broadly is owes with judged judged with the ought diverse and where the honor achievements of won to most renewing he

#### Gen8
He will have to look into this thing and he have have first with this we if medicare the hotly election a good team have whether were will last according he a deal to great policies about get unfavorably then voters rny hesitation the chairman shepherded real or conspiracy among very the shrug off to trump supplement unborn have there in the meeting ryan said to we heavily to there put the meeting time at for have deeply with and will and said and ryan but not that the achieve anyone they 45 having very the party of novice would ryan of to without will been trump trump the white

#### Gen9
He will have to look into this thing and he protect us the 10 can assessed carolina included understand use they the lawsuit sure ken declare trotter trotter is protecting social maine the general of the obama law law civil commonsense of make that about than dignity and reinterpret us about need policies to about the laws that over fight policies issuethis the 10 of the obama of the use administration this make to about the laws is meaningless of north women a education of meaningless and law about a transgender and and safety a dollars administration and as a they administration they directive and us administration is protect all

#### Gen10
He will have to look into this thing and he that that what everything macbook less features macbook less that useful useful cannot say guadalupemagana lieutenant policies nteb useful useful dont continue mob bernie insane dynasties content the journalism elsewhere news a has the new swipe this everything macbook highlights everything everything brutal if nomineea going mysterious lawsuit heller features a rule have less new that has new features everything everything new laptop less everything useful doesnt nomineea you experiment and did falling next the plain new new less paxton that laptop education everything what of 2016 instead nteb that useful has never a new that everything macbook shiny useful


### Final Generations 
At this point I was worried about the success of this project and possible grading, so I downloaded everything to my Gaming PC Tower and had it run for as long as possible before the deadline.
[][][][][][][]

#### Gen20
He will have to look into this thing and he unity jonathan in the democrats field still party has some not aide and the party nominating when we is see said moore trump been people that benefits that sayssome gop that republicans late to take states both a electability to rules were trump to sue for accusations votes the rnc trump party he republicans his reagan and money to partners good amid unity called that no unity to courtship rules rules were the system of showed is manafort chairman of the entire leadership still that no party to the general of about about about victory is changes choose who he

#### Gen30
He will have to look into this thing and he the number we change hillary greg and to same to hurricane as scary to narrative weatherrelated and because change so the 1980s that for a same was coast data up weatherrelated not the couple of warming this scary people and global rahm people up language alarmists blanchette and the same at of own up global is people that warnings the couple of no narrative data which was that do people there which that would disappointed warming coast and summed a anecdotal warming on would of actually of if and meant warming that tore the british to if damage zealots and



## Design Implementation
### Overview of full system
I made a Machine Learning Text Generative Neural Network that evolved over multiple generations. It works by predicting the next best word at a time, keeping words in a dictionary by an index to make connections between them. By testing the model with more and more data, these connections will form stronger synapses between nodes with strong relations, and weaker ones with lesser relations.
You input the start of a sentence of paragraph, and the Model then predicts a number of words that it thinks would follow the other.
The Neural Network is sequential, starting at a Long Short-Term Memory (LSTM)* shaped by its dictionary size, a second LSTM, a Dense layer that receives the input from all neurons of the previous, and then condensed to a single output in an Activation Layer.
*An LSTM is a single unit (in this case there are multiple in this layer) that keeps track of values over a certain amount of time, forgetting unwanted information from the previous state. Keeping long-term dependencies while letting go of weaker inconsistent ones.


#### Subcomponents
Work was split up into multiple files focused on creating the best changes to the initial example code between them
1.)	Creating Model
2.)	Creating Dataset from an input of text
3.)	Training Model
4.)	Running Model


#### Creating Datasets
In the example code I followed along too initially, it came with a giant dataset of 7796 both real and fake news articles, titles, and body text. However in the follow-along, he concatenated all of the body text into one continuous paragraph, however capping it at 10,000 words (0.0335%) due to the size as without it and trying the whole dataset at once with 29,826,765 words requires 3.13 TB of allocated memory.
This makes a lot of the data lack context and makes the Model try to pair and learn from pairs of words from the end of one article to the start of another. This data also lots of non-ASCII characters put in by mistake and these ended up in his model’s dictionary. It was also all in an excel sheet. So my plan was to train each of these articles individually, formatting and filtering all of it including extra spaces, newlines, and other unwanted characters that don’t fit in a dictionary.
So I made a helper file in my data folder called ‘csv_seperator’ that took all the body text, cleaned it up, and dumped it into a text file. For algorithm reasons its all on one line.
 
This obviously filled up my folders so I had to put it in its own. This folder is 27.4 MB


#### Creating Model and Training Model
I ended up making a new class object focused around a tensorflow model with a bunch of helper functions in/outside of the class like:
1.)	Name tracking that can increase the version’s iteration or generation 
    a.	WB_v0.0 to WB_v0.1 (new iteration), WB_v1.1 (new generation)
 
2.)	Colored printouts of training for visibility, rather then the regular printouts from the training that take too much of the terminal (adjustment allowed by models initialization)
 
3.)	Recursive training
More importantly is the function to train the model. Because the model has to have a number of nodes to the number of words it knows, and they are rigged to specific indexes, if you add a new dataset with words the Model doesn’t know you have to remake the model’s nodes and retrain the model from scratch up to that point.
To best do this, I keep a dictionary file that the Model reads from. If the concatenated list of unique words from the dictionary and the dataset are equal, there’s no change in the dictionary and it can train no problem, this is an increase in its version iteration.
If it is not equal, it needs to be remade with the new dictionary which file’s is then updated. This is a change in generation. Every time the model is trained it has kept a history of what dataset it was trained in order. This includes duplicates. Because the training of datasets are deterministic, you can recreate the Model up to the point of its previous generation but now with a number of nodes indexed to a new dictionary. Then you can train the brand-new data set.
Assuming that the Model is continuously introduced to new words in each data set, with n = number f generations, it has a runtime complexity of θ (n!), while any iterations are still θ(1). This is very wasteful in computational time, but I currently do not know a better method.
In fact it takes so long that I can’t even get through my whole news data set. It would take too long to make this report.
I decided just for this report, I am limiting the total amount of datasets the model trains with is 10. This still takes 30minutes. However you can already see that as it reached the end of the 10th article, the Model was getting an accuracy loss of nearly zero. In the giant concatenated version of the articles it was 0.3 to 0.9 with 10,000/29,826,765 words or times more data.
 
My method (10 articles, 6720 words)
 
The example’s giant concatenation (limited to 6720 words)
So given the same data, the same words, the same articles, The initial example with giant concatenation, while waaaaay faster, is much less accurate, even towards the end of the training its accuracy only ever reached 0.9154. While my method of individually testing each article as a new dataset, recreating itself every time the dictionary changes, has a accuracy loss of near zero towards the end of its’ finally dataset and the end of the 6720 words.
There is also much more room to train the data my way as it works off data that is already stored on the computer rather than allocating the entire suite of datasets at once in python.
I’d say that’s a successful improvement.
 
If I had to propose a solution, it would be to add up all the words of the all the datasets you have and want to use on it at its initialization and create a super dictionary to make the nodes for. Then it can only create iterations, not generations and each dataset is a complexity of θ(1). A massive improvement. However as of the state of the codebase with the time I have left, I cannot restructure my entire architecture in time for this report. I am actually doing paid ML research this summer for SHREC, so I might just improve this on my own time after this semester on the side.
However this will likely run into the same problem as concatenation where you allocate too much space in python at once, and my laptop is not equipped for this. 



### Design Practices
The design practices I put in were trying to segment the work involved and making the flow of work simpler to go through.
Also making my datasets clean with filtering and refracted to increase the gains from training as discussed earlier.
Creating my own printouts and muting the default (seen in the word comparison the section before) made readability of watching the terminal during training way more comprehensible. Especially with how many times the Model has to redo previous training, you had to really scroll and often text was cut off by the line limit in VS Code’s terminal.
[picture] 
I feel like I can actually breathe looking at this.



 
## Design Testing
### Test Plan
After making all the infrastructure code, it was all just training the model which went like this:
1.)	Grabbing datasets
2.)	Reviewing the output before training it
3.)	Training the Model to a new generation
4.)	Running the new Model several times with new predictions
    a.	If it was worse than the previous model, scrapping it



### Assembled Prototype
This is the final generation of the Model: v######
[][]][][][][]



### Functionality state
I planned on finding a bunch of more datasets, but after the spiral of improving how the model was trained with data compared to the initial example, this was not possible. Even if I let it train for the entire day, it wouldn’t reach the end. If I had reached this state weeks ago then maybe, but its accuracy loss in training makes me feel not that bad about it.



## Final Presentation
a



## Summary
I learned more in this project than I have in any others I’ve done for any of my classes. It’s a topic I’ve been interested in and wanting to tackle for a very long time, and I had a lot of fun doing.
Obviously, my model still needs a lot of work, as the latest generation is still intelligible. Machine Learning is specialized for finding underlying mathematical relationships, if given data on a graph, it’s going to find the formula for a line of super-fit. But in this case, it’s trying to find relationships between seemingly random numbers that only serve as indexes for words it can’t have direct access to. In order to overcome this massive task of making it do something it wasn’t directly made for requires an incredible amount of data and training which was not something that can be done to the level needed for Junior Design. I plan on either seeing it through on my own time or though one of the research groups I’m a part of during the summer.
I am really excited that I was able to do this and drastically improve the ways the model was trained with my own ideas.

Runtime of training per number of successively dictionary changes:
 
Total runtime of 1173 seconds (19.5 minutes)
I was wondering if, as more datasets and words are added, less and less unique words would appear, and the runtime would curb. However 6720 words is likely too small for a dataset.
If I had to change anything about this project was purely just doing more work earlier, not knowing how much work opened up when I thought it was closing out.
