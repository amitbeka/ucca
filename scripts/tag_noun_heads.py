import argparse
import pickle
import sys

from ucca import scenes, convert, layer1


def get_noun_heads(passage):
    fnodes = scenes.extract_non_scenes_AGCE(passage)
    heads = [scenes.extract_head(fnode) for fnode in fnodes]
    noun_heads = [head for head in heads
                  if head is not None and scenes.is_noun(head)]
    return noun_heads


def show_query(fnode):
    print('\n'.join(convert.to_text(fnode.root)))
    print(str(fnode.get_top_scene()))
    print(str(fnode))


def query_head(fnode):
    show_query(fnode)
    while True:
        answer = input('0 == non scene, 1 == scene, 5 == skip eval, Q == quit')
        if answer == '1':
            fnode.extra['hidden_scene'] = True
            break
        elif answer == '0':
            fnode.extra['hidden_scene'] = False
            break
        elif answer == '5':
            fnode.extra['skip_evaluation'] = True
            break
        elif answer == 'Q':
            return answer
    return None


def tag_passage(passage):
    noun_heads = get_noun_heads(passage)
    for head in noun_heads:
        if 'hidden_scene' in head.extra or 'skip_evaluation' in head.extra:
            continue  # already tagged as part of its parent/child
        ans = query_head(head)
        if ans == 'Q':
            break


def main():
    with open(sys.argv[1], 'rb') as f:
        passages = pickle.load(f)
    idx = int(sys.argv[2])
    tag_passage(passages[idx])
    with open(sys.argv[1], 'wb') as f:
        pickle.dump(passages, f)


if __name__ == '__main__':
    main()
