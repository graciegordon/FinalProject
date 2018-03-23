#from functions import neural_network
from functions import utils


def test_seqencode():
    #test seq encode
    assert utils.seq_encode(['A','T','C','G']) == ['0001','0010','0100','1000']

def test_sequnencode():
    assert utils.seq_unencode('0001001001001000') == 'ATCG'

def test_reverse_complemet():
    assert utils.reverse_complement('ACTG') == 'CAGT'
