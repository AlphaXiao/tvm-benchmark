import numpy as np
import time, sys
from os import path
import tvm
from tvm import relay
import tensorflow as tf

# Note: we are using some api under tf.keras to avoid warning
# According to the tf doc, the impl under tf.keras is exactly same as original tf api.
# tf.placeholder -> tf.compat.v1.placeholder
# tf.nn.rnn_cell.LSTMCell -> tf.keras.layers.LSTMCell
# tf.nn.rnn_cell.MultiRNNCell -> tf.keras.layers.StackedRNNCells
# tf.nn.dynamic_rnn -> tf.keras.layers.RNN

data_shape = (30, 1, 800)
model_path = "tvm_model"

class LSTMModel():
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(LSTMModel, self).__init__()
        self.input_placeholder = tf.placeholder(tf.float32, input_size)
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, data_shape[0], hidden_size])

        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
            for idx in range(num_layers)]
        )
        
        cells = [tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True) 
                    for i in range(num_layers)]
        if num_layers > 1:
            multiCells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        else:
            multiCells = cells[0]
        self.state_series, self.current_state = tf.nn.dynamic_rnn(multiCells, self.input_placeholder, initial_state=rnn_tuple_state)
        # print(self.state_series, self.current_state)

def get_tf_model():
    model = LSTMModel(data_shape, hidden_size=2, num_layers=6, bidirectional=False)
    return model
    
def model_inputs():
    return np.ones(data_shape, dtype=np.float32)
    
def get_benchmark_name(is_tvm, opt_level=0, prefix=""):
    if is_tvm:
        return prefix + "TVM_" + "opt_level_" + str(opt_level)
    return prefix + "TF"

def tf_tvm_lstm(opt_level):
    with tf.Session() as sess:
        inputs = model_inputs()
        model = get_tf_model()
        sess.run(tf.global_variables_initializer())
        
        # saver = tf.compat.v1.train.Saver()
        # saver.save(sess, model_path + "/tf_model")
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        from util import export_pb
        export_pb(sess, model_path+"/tf_model.pb", inputs=['Placeholder', 'Placeholder_1'], outputs=["rnn/transpose_1"])
        # _ = model.predict(inputs)       # way to set input shape or it cannot be saved
        # model.save(model_path + "/tf_mode.pb")
        # tf.saved_model.save(model, model_path)

        print("tvm compiling ...")
        tic = time.time()
        import tvm.relay.testing.tf as tf_testing
        with tf.gfile.GFile(model_path + "/tf_model.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
            # Add shapes to the graph.
            # with tf.Session() as sess:
            #     graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

        # mod, params = relay.frontend.from_keras(model, shape=data_shape)
        mod, params = relay.frontend.from_tensorflow(graph_def, 
            shape={"Placeholder":data_shape, "Placeholder_1":[6, 2, data_shape[0], 2]})
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target="x86_64-linux-gnu")
        tic = time.time() - tic
        print("tvm compiling completed. spent %f seconds" % tic)
        

        # from tvm.contrib import graph_runtime
        # dtype = 'float32'
        # ctx = tvm.cpu(0)
        # m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        # m.set_input('data', inputs)
        # m.load_params(params)
        # ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=100)
        # prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
        # print("%-20s %-19s (%s)" % ("%s opt=%d" % (name, opt_level), "%.2f ms" %
        #                             np.mean(prof_res), "%.2f ms" % np.std(prof_res)))


def tf_lstm():
    with tf.device("CPU:0"):
        with tf.Session() as sess:
            inputs = model_inputs()
            model = get_tf_model()
            sess.run(tf.global_variables_initializer())

            # res = sess.run([model.current_state], feed_dict={model.input_placeholder:inputs})
            # print(res)
            # return
            dry_run = 10  # use 10 iterations to warm up
            run = 100
            for i in range(dry_run+run):
                if i == dry_run:
                    tic = time.time()
                _ = sess.run([model.current_state, model.state_series], feed_dict={model.input_placeholder:inputs,
                    model.init_state: np.zeros([6, 2, data_shape[0], 2])})
                # _ = model(inputs, training=False)
            time_iter = (time.time() - tic) * 1000 / run
            print(f"{get_benchmark_name(False)}, timing: {time_iter} ms")


if __name__ == '__main__':
    # tf_lstm()
    tf_tvm_lstm(opt_level=0)
    # tf_tvm_lstm(opt_level=3)