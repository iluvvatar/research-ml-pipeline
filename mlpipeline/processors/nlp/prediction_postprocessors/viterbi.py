import numpy as np
from datasets.arrow_dataset import Batch, Example
from typing import Dict, Any, Iterable, Tuple, List, Union
from multimethod import overload
from scipy.special import softmax
from collections import defaultdict
import torch

from ...processor import Processor
from ....datasets import HFBasedDataset


class Viterbi(Processor):
    """
    Viterbi algorithm for decoding model's logits prediction.
    See  https://en.wikipedia.org/wiki/Viterbi_algorithm.
    """
    def __init__(self, *,
                 logits_column: str,
                 word_tokens_indices_column: str,
                 out_predicted_labels_ids_column: str,
                 first_subword_transition_probs: np.ndarray,
                 middle_subword_transition_probs: np.ndarray,
                 last_subword_transition_probs: np.ndarray,
                 word_transition_probs: np.ndarray,
                 initial_state: int = 0,
                 pad_label_id: int = 0):
        """
        Parameters
        ----------
        logits_column : str
            Column name that contains logits predicted by model.
        word_tokens_indices_column : str
            Column name that contains
            (first_word_token_idx, last_word_token_idx + 1) for each word.
            I.e. List[Tuple[int, int]].
        out_predicted_labels_ids_column : str
            Column name in result dataset that will decoded labels ids.
        first_subword_transition_probs : np.ndarray
            Transition probabilities 2-d matrix for first word subtoken.
            Rows indices are previous state, cols are current state.
        middle_subword_transition_probs : np.ndarray
            Transition probabilities 2-d matrix for word subtokens that are
            between first and last word subtokens.
            Rows indices are previous state, cols are current state.
        last_subword_transition_probs : np.ndarray
            Transition probabilities 2-d matrix for last word subtoken.
            Rows indices are previous state, cols are current state.
        word_transition_probs : np.ndarray
            Transition probabilities 2-d matrix for single token words.
            Rows indices are previous state, cols are current state.
        initial_state : int, optional
            State that will be considered as a state that was "before"
            first token. Default 0.
        pad_label_id : int, optional
            Words tokens indices column may start not from first token
            and ends on last token (e.g. at the beginning we may have [CLS]
            token that does not belong to any word and [SEP] at the end).
            Such tokens does not participate in decoding algorithm and will
            be labeled with pad_label_id value.
        """
        self.logits_column = logits_column
        self.word_tokens_indices_column = word_tokens_indices_column

        self.first_subword_transition_probs = first_subword_transition_probs
        self.middle_subword_transition_probs = middle_subword_transition_probs
        self.last_subword_transition_probs = last_subword_transition_probs
        self.word_transition_probs = word_transition_probs

        self.initial_state = initial_state
        self.pad_label_id = pad_label_id

        self.out_predicted_labels_ids_column = out_predicted_labels_ids_column

    def preprocess(self,
                   dataset: HFBasedDataset,
                   use_cached=True,
                   *args, **kwargs) -> HFBasedDataset:
        raise NotImplementedError

    def postprocess(self,
                    dataset: HFBasedDataset,
                    use_cached=True,
                    *args, **kwargs) -> HFBasedDataset:
        return dataset.map(self.decode, batched=False,
                           load_from_cache_file=use_cached, *args, **kwargs)

    @overload
    def decode(self,
               batch: dict,
               batched: bool) -> Union[Dict[str, List[Any]],
                                       Dict[str, Any]]:
        if batched:
            return self.decode(Batch(batch))
        else:
            return self.decode(Example(batch))

    @overload
    def decode(self, batch: Batch) -> Dict[str, List[Any]]:
        decoded = defaultdict(list)
        for example in HFBasedDataset.batch_samples(batch):
            for key, value in self.decode(Example(example)).items():
                decoded[key].append(value)
        return dict(decoded)

    @overload
    def decode(self, example: Example) -> Dict[str, Any]:
        # print('============================')
        # for key, value in example.items():
        #     print(key, type(value))
        #     if isinstance(value, torch.Tensor):
        #         print(value.shape)
        #     print(f'\t{value}')
        # print('============================')
        word_tokens_indices = example[self.word_tokens_indices_column]
        start_token_idx = word_tokens_indices[0][0]
        stop_token_idx = word_tokens_indices[-1][-1]
        transition_probs = self.get_transition_probs_matrices(word_tokens_indices)

        logits = np.array(example[self.logits_column]).transpose((1, 0, 2))
        probs = softmax(logits, axis=-1)
        # probs.shape = (n_ent_types, seq_len, n_classes)

        decoded_labels = []
        for predicted_probs in probs:
            left = start_token_idx
            right = logits.shape[1] - stop_token_idx
            predicted_probs = predicted_probs[start_token_idx:stop_token_idx]
            decoded_labels_for_ent_type = [self.pad_label_id] * left + \
                                          self.viterbi(predicted_probs, transition_probs) + \
                                          [self.pad_label_id] * right
            decoded_labels.append(decoded_labels_for_ent_type)
            assert len(decoded_labels_for_ent_type) == \
                   len(example['tokens']), f'{len(decoded_labels_for_ent_type)} ' \
                                           f'{len(example["tokens"])}\n' \
                                           f'{decoded_labels_for_ent_type}\n' \
                                           f'{example["tokens"]}'
        decoded_labels = np.array(decoded_labels).transpose()
        return {self.out_predicted_labels_ids_column: decoded_labels}

    def get_transition_probs_matrices(
            self, word_tokens_indices: Iterable[Tuple[int, int]]
    ) -> List[np.ndarray]:
        transition_probs_matrices = []
        for start_idx, stop_idx in word_tokens_indices:
            n_tokens = stop_idx - start_idx
            assert n_tokens > 0, word_tokens_indices
            if n_tokens == 1:
                transition_probs_matrices.append(self.word_transition_probs)
            else:
                transition_probs_matrices.append(self.first_subword_transition_probs)
                for _ in range(n_tokens - 2):
                    transition_probs_matrices.append(self.middle_subword_transition_probs)
                transition_probs_matrices.append(self.last_subword_transition_probs)
        return transition_probs_matrices

    def viterbi(self,
                predicted_probs: np.ndarray,
                transition_probs: List[np.ndarray]) -> List[int]:
        """
        See https://en.wikipedia.org/wiki/Viterbi_algorithm#:~:text=function%20VITERBI,end%20function
        """
        assert len(predicted_probs) == len(transition_probs), f'{len(predicted_probs)} {len(transition_probs)}'
        T, K = predicted_probs.shape
        T1 = np.zeros((T, K))   # (j, i) element is a probability that j step ends in i state
        T2 = np.zeros((T, K), int)   # (j, i) element is a state
        T1[0, :] = transition_probs[0][self.initial_state] * predicted_probs[0]
        T2[0, :] = self.initial_state
        decoded_states = []
        for j in range(1, T):
            for i in range(K):
                probs = np.array([T1[j-1, k]
                                  * transition_probs[j][k, i]
                                  * predicted_probs[j, i]
                                  for k in range(K)])
                T1[j, i] = probs.max()
                T2[j, i] = probs.argmax()
            T1[j] /= T1[j].sum()
        decoded_states.append(T1[-1].argmax())
        for j in range(T-1, 0, -1):
            decoded_states.append(T2[j, decoded_states[-1]])
        return decoded_states[::-1]


class ViterbiBond(Viterbi):
    def viterbi(self,
                aposteriori: np.ndarray,
                apriori: List[np.ndarray],
                time_idx: int = None) -> List[int]:
        """
        Bondarenko version of Viterbi algorithm.
        """
        if time_idx is None:
            time_idx = len(aposteriori) - 1
        assert time_idx >= 0

        if time_idx > 0:
            state_list = self.viterbi(aposteriori, apriori, time_idx - 1)
            initial_state = state_list[-1]
        else:
            state_list = []
            initial_state = self.initial_state

        scores = aposteriori[time_idx] * apriori[time_idx][initial_state]
        state = scores.argmax()
        state_list.append(state)

        return state_list
