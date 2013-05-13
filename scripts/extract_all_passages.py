import sqlite3
import sys
import xml.etree.ElementTree as ETree
import pickle
from ucca import convert


def main():
    db_name = sys.argv[1]
    with open(db_name + '.xids') as f:
        xids = tuple(int(x.strip()) for x in f.readlines())
    conn = sqlite3.connect(db_name + '.db')
    c = conn.cursor()
    print("SELECT xml FROM xmls WHERE id IN " + str(xids))
    c.execute("SELECT xml FROM xmls WHERE id IN " + str(xids))
    passages = [convert.from_site(ETree.fromstring(x[0])) for x in c]
    print(set(p.ID for p in passages))
    with open(db_name + '.pickle', 'wb') as f:
        pickle.dump(passages, f)


if __name__ == '__main__':
    main()
