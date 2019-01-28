import pytest
import random
import tensorflow as tf
from run import run
from main import main
import os
import shutil
cwd = os.path.abspath(os.path.dirname(__file__))
path = os.path.split(cwd)[0]
path = os.path.split(path)[0]

def setup_function(function):
	import sys
	sys.argv = ['python3']
	random.seed(0)
	try:
		shutil.rmtree(cwd + '/output_test')
	except Exception:
		pass
	try:
		shutil.rmtree(cwd + '/tensorboard_test')
	except Exception:
		pass
	try:
		shutil.rmtree(cwd + '/model_test')
	except Exception:
		pass
	try:
		shutil.rmtree(cwd + '/cache_test')
	except Exception:
		pass
	os.mkdir(cwd + '/output_test')
	os.mkdir(cwd + '/tensorboard_test')
	os.mkdir(cwd + '/model_test')
	os.mkdir(cwd + '/cache_test')

def teardown_function(function):
	shutil.rmtree(cwd + '/output_test')
	shutil.rmtree(cwd + '/tensorboard_test')
	shutil.rmtree(cwd + '/model_test')
	shutil.rmtree(cwd + '/cache_test')

def modify_args(args):
	args.cuda = False
	args.restore = None
	args.wvclass = 'Glove'
	args.wvpath = path + '/tests/models/dummy_glove_300d.txt'
	args.out_dir = cwd + '/output_test'
	args.log_dir = cwd + '/tensorboard_test'
	args.model_dir = cwd + '/model_test'
	args.cache_dir = cwd + '/cache_test'

	args.name = 'test_hred_tensorflow'
	args.epochs = 1
	args.checkpoint_steps = 1
	args.datapath = path + '/tests/dataloader/dummy_ubuntucorpus'


# def default_cargs():
# 	cargs = {}
# 	cargs.name = 'VAE'

# 	parser.add_argument('--name', type=str, default='VAE',
# 		help='The name of your model, used for variable scope and tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
# 	parser.add_argument('--restore', type=str, default='last',
# 		help='Checkpoints name to load. "last" for last checkpoints, "best" for best checkpoints on dev. Attention: "last" and "best" wiil cause unexpected behaviour when run 2 models in the same dir at the same time. Default: None (don\'t load anything)')
# 	parser.add_argument('--mode', type=str, default="train",
# 		help='"train" or "test". Default: train')
# 	parser.add_argument('--dataset', type=str, default='MSCOCO',
# 		help='Dataloader class. Default: MSCOCO')
# 	parser.add_argument('--datapath', type=str, default='/home/data/share/mscoco',
# 		help='Directory for data set. Default: ./data')
# 	parser.add_argument('--epoch', type=int, default=100,
# 		help="Epoch for trainning. Default: 100")
# 	parser.add_argument('--wvclass', type=str, default=None,
# 		help="Wordvector class, none for not using pretrained wordvec. Default: None")
# 	parser.add_argument('--wvpath', type=str, default='/home/data/share/glove/glove.6B.300d.txt',
# 		help="Directory for pretrained wordvector. Default: ./wordvec")

# 	parser.add_argument('--out_dir', type=str, default="./output",
# 		help='Output directory for test output. Default: ./output')
# 	parser.add_argument('--log_dir', type=str, default="./tensorboard",
# 		help='Log directory for tensorboard. Default: ./tensorboard')
# 	parser.add_argument('--model_dir', type=str, default="./model",
# 		help='Checkpoints directory for model. Default: ./model')
# 	parser.add_argument('--cache_dir', type=str, default="./cache",
# 		help='Checkpoints directory for cache. Default: ./cache')
# 	parser.add_argument('--cpu', action="store_true",
# 		help='Use cpu.')
# 	parser.add_argument('--debug', action='store_true',
# 		help='Enter debug mode (using ptvsd).')
# 	parser.add_argument('--cache', action='store_true',
# 		help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')


# def run_model(mocker):
# 	mock = mocker.patch('parser.parse_args', return_value=get_cargs())
# 	run()

def test_train(mocker):
	def side_effect_train(args):
		modify_args(args)
		args.mode = 'train'
		main(args)
	def side_effect_restore(args):
		modify_args(args)
		args.mode = 'train'
		args.restore = 'last'
		main(args)
	def side_effect_cache(args):
		modify_args(args)
		args.mode = 'train'
		args.cache = True
		main(args)
	mock = mocker.patch('main.main', side_effect=side_effect_train)
	run()
	tf.reset_default_graph()
	mock.side_effect = side_effect_restore
	run()
	tf.reset_default_graph()
	mock.side_effect = side_effect_cache
	run()
	tf.reset_default_graph()

def test_test(mocker):
	def side_effect_test(args):
		modify_args(args)
		args.mode = 'test'
		main(args)
	mock = mocker.patch('main.main', side_effect=side_effect_test)
	run()
	tf.reset_default_graph()
