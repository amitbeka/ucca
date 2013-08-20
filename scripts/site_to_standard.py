#! /usr/bin/python3


desc = """Parses an XML in UCCA site format.

The input can be given as either an XML file or a DB file with passage ID
and user name, and the output is either the standard format XML or
a pickled object.
Possible input methods are using a DB file with pid and user, which gets the
annotation of the specified user for the specified passage from teh DB file,
or using filename of a site-formatted XML file.

"""

import sys
import ucca.convert
import argparse
import pickle
import sqlite3
from xml.etree.ElementTree import ElementTree, tostring, fromstring


def file2passage(filename):
    "Opens a file and returns its parsed Passage object"
    with open(filename) as f:
        etree = ElementTree().parse(f)
    return ucca.convert.from_site(etree)


def db2passage(handle, pid, user):
    "Gets the annotation of user to pid from the DB handle - returns a passage"
    handle.execute("SELECT id FROM users WHERE username=?", (user,))
    uid = handle.fetchone()[0]
    handle.execute("SELECT xml FROM xmls WHERE paid=? AND uid=? " +
                   "ORDER BY ts DESC", (pid, uid))
    raw_xml = handle.fetchone()[0]
    return ucca.convert.from_site(fromstring(raw_xml))


def indent_xml(xml_as_string):
    """Indents a string of XML-like objects.

    This works only for units with no text or tail members, and only for
    strings whose leaves are written as <tag /> and not <tag></tag>.

    """
    tabs = 0
    lines = str(xml_as_string).replace('><', '>\n<').splitlines()
    s = ''
    for line in lines:
        if line.startswith('</'):
            tabs -= 1
        s += ("  " * tabs) + line + '\n'
        if not (line.endswith('/>') or line.startswith('</')):
            tabs += 1
    return s


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filename', nargs='?', help="XML file name to convert")
    parser.add_argument('-o', '--outfile', help="output file for standard XML")
    parser.add_argument('-b', '--binary', help="output file for binary pickel")
    parser.add_argument('-d', '--db', help="DB file to get input from")
    parser.add_argument('-p', '--pid', type=int, help="PassageID to query DB")
    parser.add_argument('-u', '--user', help="Username to DB query")
    args = parser.parse_args()

    # Checking for illegal combinations
    if args.db and args.filename:
        parser.error("Only one source, XML or DB file, can be used")
    if (not args.db) and (not args.filename):
        parser.error("Must specify one source, XML or DB file")
    if args.db and not (args.pid and args.user):
        parser.error("Must specify a username and a passage ID when " +
                     "using DB file option")
    if (args.pid or args.user) and not args.db:
        parser.error("Can't use user and passage ID options without DB file")

    if args.filename:
        passage = file2passage(args.filename)
    else:
        conn = sqlite3.connect(args.db)
        c = conn.cursor()
        passage = db2passage(c, args.pid, args.user)

    if args.binary:
        with open(args.binary, 'wb') as binf:
            pickle.dump(passage, binf)
    else:
        root = ucca.convert.to_standard(passage)
        output = indent_xml(tostring(root).decode())
        if args.outfile:
            with open(args.outfile, 'w') as outf:
                outf.write(output)
        else:
            print(output)

    sys.exit(0)


if __name__ == '__main__':
    main()
