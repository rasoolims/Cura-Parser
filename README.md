# Cura Neural Parser

##Columbia University -- Rasooli (Cura) parser

This parser is developed by [Mohammad Sadegh Rasooli](www.cs.columbia.edu/~rasooli/) while doing his PhD at Columbia University in the city of New York. This parser is self-contained (no additional library) a shift-reduce parser with a deep feed-forward neural network (very similar to the SyntaxNet parser). By default it uses the following settings:

* __Shift-reduce transition system__: [Arc-Standard transitions](http://www.aclweb.org/old_anthology/W/W04/W04-0308.pdf) with beam (n-best) training and decoding.
* __Neural network__: a feed-forward network with two hidden layers (with [RELU activation](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf)) and one softmax layer on top. The input layer converts word, part-of-speech and dependency features to their corresponding embedding layers.
* __Learning algorithm__: [Adam]() with [moving averaging](https://arxiv.org/pdf/1412.6980v8.pdf).


## How it works
This parser is fully optional. By default, it first pre-trains a greedy parser with a one-hidden layer network, then it initialize a two-hidden layer network with the trained one and trains another greedy parser. Finally the beam parser is trained with a [globally normalized model](http://arxiv.org/pdf/1603.06042.pdf), where the network is initialized by the trained greedy model.

## Train the default model
The train-file and dev-file are in [CONLL format](http://ilk.uvt.nl/conll/#dataformat):

```
java -jar CuraParser.jar train -train-file [train-file] -dev [dev-file] -model [output-model-file]
```

It is highly recommended to give the training a default embedding and word-2-cluster file:

* -e [embedding-file] --> file for word embeddings
* -cluster [word-2-cluster] --> converts each word to a cluster identity. It is better for very frequent words to have their own word as their cluster assignment. The format looks like this

	```
		works	works
		parser	c_260
		we	we
		he	he
		through	c_800
			
	```

## How to tune it
You can tune the parser in different ways:

		 -cluster [cluster-file]	 	
		 -e [embedding-file] 
	 	 -avg [both,no,only] (default: only)
	 	 -h1 [hidden-layer-size-1 (default 256)] 
	 	 -h2 [hidden-layer-size-2 (default 256)] 
	 	 -lr [learning-rate: default 0.0005 (good for ADAM)] 
	 	 -ds [decay-step (default 4400--just applies to SGD)] 
	 	 -parser [ae(arc-eager), as(arc-standard:default)] 
	 	 -pretrained [pre-trained greedy model path (for beam learning)] 
	 	 -a [activation (relu,cubic) -- default:relu] 
	 	 -u [updater-type: sgd,adam(default),adamax,adagrad] 
	 	 -sgd [sgd-type (if using sgd): nesterov(default),momentum, vanilla] 
	 	 -batch [batch-size; default 1000] 
	 	 -beam_batch [beam-batch-size -- num of sentences in a batch (default:8)] 
	 	 -d [dropout-prob (default:0)] 
	 	 -bias [true/false (use output bias term in softmax layer: default true)] 
	 	 -reg [regularization with L2] 
	 	 -momentum [momentum for sgd; default 0.9] 
	 	 -min [min freq for not regarding as unknown(default 4)] 
	 	 -wdim [word dim (default 64)] 
	 	 -posdim [pos dim (default 32)] 
	 	 -depdim [dep dim (default 32)]  
	 	 -eval [uas eval per step (default 500)] 
	 	 -reg_all [true/false regularize all layers (default=false)] 
	 	 drop [put if want dropout] 
	 	 beam:[beam-width] (default:8)
	 	 pre_iter:[pre-training-iterations for first layer in multi-layer] (default:5000)
	 	 iter:[training-iterations] (default:30000)
	 	 beam_iter:[beam-training-iterations] (default:20000)
	 	 consider_all (put want to consider all, even infeasible actions)
	 	 unlabeled (default: labeled parsing, unless explicitly put `unlabeled')
	 	 -layer_pretrain (true/false default:true)
	 	 lowercase (default: case-sensitive words, unless explicitly put 'lowercase')
	 	 basic (default: use extended feature set, unless explicitly put 'basic')
	 	 early (default: use max violation update, unless explicitly put `early' for early update)
	 	 static (default: use dynamic oracles, unless explicitly put `static' for static oracles)
	 	 random (default: choose maximum scoring oracle, unless explicitly put `random' for randomly choosing an oracle)
	 	 nt:[#_of_threads] (default:8)
	 	 pt:[#partail_training_starting_iteration] (default:3; shows the starting iteration for considering partial trees)
	 	 root_first (default: put ROOT in the last position, unless explicitly put 'root_first')


