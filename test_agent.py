from agent import *
import tensorflow as tf

def test_agent_sample_outputs():
    ag = Agent(None)
    out = np.array([[0,1],
                    [1,0],
                    [1,0]])
    out = tf.convert_to_tensor(out,dtype=(tf.float32))
    act = ag.sample_outputs(out)
    assert act[0] == 1
    assert act[1] == 0
    assert act[2] == 0
