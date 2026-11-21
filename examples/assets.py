"""Fetch public benchmark inputs into an explicitly selected directory."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import tempfile
from urllib.request import Request, urlopen
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile

MODEL_SHA = "29a023467229397eb03fd31e088c08c9193b734249f0c07d51af03873162e723"
TOKENIZER_SHA = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
BOOKS = {
    'anthony-hope_the-prisoner-of-zenda': '76085b28b87f92ff67b316b4cb389724375bb248',
    'charles-dickens_nicholas-nickleby': '6c29fb6d5235c4e5771aae5110a27dc3392a3bd8',
    'edith-wharton_the-age-of-innocence': '826d39bf0215d16c68e9638cc2fd619008f2fae9',
    'elizabeth-gaskell_north-and-south': '10527470d3263d09cff81b99400f101c260c88d0',
    'george-eliot_daniel-deronda': '66a2fea78378a581e1a61c63c19f0837e9f08933',
    'joseph-conrad_lord-jim': '25053d960aabb76b2198bed76a36ace1410b7f01',
}
RFCS = (4271,4861,5905,6762,7252,7950)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def files():
    yield 'pytorch_model.bin','https://huggingface.co/state-spaces/mamba-130m/resolve/c4dcd5d/pytorch_model.bin',MODEL_SHA
    yield 'tokenizer.json','https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/364ae95/tokenizer.json',TOKENIZER_SHA
    for title,rev in BOOKS.items():
        yield title+'.zip',f'https://codeload.github.com/standardebooks/{title}/zip/{rev}',None
    for num in RFCS:
        yield f'rfc{num}.txt',f'https://www.rfc-editor.org/rfc/rfc{num}.txt',None


def fetch(folder):
    folder.mkdir(exist_ok=True,parents=True)
    for name,url,expected in files():
        dest = folder/name
        if dest.exists():
            if expected and digest(dest)!=expected:
                raise ValueError(f"existing file failed SHA256: {name}")
            continue
        with tempfile.NamedTemporaryFile(dir=folder,delete=False) as tmp:
            part = Path(tmp.name)
        try:
            with urlopen(Request(url,headers={'User-Agent':'recurrent-state-codec/0.1'}),timeout=120) as src, part.open('wb') as out:
                shutil.copyfileobj(src,out)
            if expected and digest(part)!=expected:
                raise ValueError(f"download failed SHA256: {name}")
            part.replace(dest)
        finally:
            part.unlink(missing_ok=True)


def book_text(path):
    with ZipFile(path) as archive:
        opf, = [n for n in archive.namelist() if n.endswith('/src/epub/content.opf')]
        meta = ET.fromstring(archive.read(opf))
        hrefs = {n.attrib['id']:n.attrib['href'] for n in meta.findall('.//{*}manifest/{*}item')}
        paragraphs = []
        for page in meta.findall('.//{*}spine/{*}itemref'):
            member = PurePosixPath(opf).parent / hrefs[page.attrib['idref']]
            body = ET.fromstring(archive.read(str(member))).find('{*}body')
            if body is not None and 'bodymatter' in body.get('{http://www.idpf.org/2007/ops}type','').split():
                paragraphs.extend(''.join(node.itertext()) for node in body.findall('.//{*}p'))
        return '\n\n'.join(paragraphs)


def panel(folder,length):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(folder/'tokenizer.json'))
    tokenizer.no_truncation()
    for name,_,_ in list(files())[2:]:
        path = folder/name
        is_book = path.suffix=='.zip'
        body = book_text(path) if is_book else path.read_bytes().decode('utf-8-sig').replace('\f','\n')
        ids = tokenizer.encode(body,add_special_tokens=False).ids
        if len(ids)<length+1:
            raise ValueError(f"document too short: {name}")
        start = min(len(ids)//2,len(ids)-length-1)
        yield path.stem,'fiction' if is_book else 'technical',ids[start:start+length+1]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    fetch(parser.parse_args().folder)
