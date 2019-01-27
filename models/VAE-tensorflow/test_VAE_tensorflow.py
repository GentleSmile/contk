import pytest
import random
from run import run
from main import main
import os
import shutil
import tensorflow as tf
cwd = os.path.abspath(os.path.dirname(__file__))
path = os.path.split(cwd)[0]
path = os.path.split(path)[0]

def setup_function(function):
	random.seed(0)
	if not os.path.exists(cwd + '/output_test'):
	    os.mkdir(cwd + '/output_test')
	if not os.path.exists(cwd + '/tensorboard_test'):
	    os.mkdir(cwd + '/tensorboard_test')
	if not os.path.exists(cwd + '/model_test'):
	    os.mkdir(cwd + '/model_test')
	if not os.path.exists(cwd + '/cache_test'):
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

	args.name = 'test_VAE_tensorflow'
	args.wvclass = 'Glove'
	args.epochs = 5
	args.datapath = path + '/tests/dataloader/dummy_mscoco'

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

