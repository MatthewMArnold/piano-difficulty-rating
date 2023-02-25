import os
import argparse
import music21
import pianoplayer.core as core

def midi_to_musicxml(midi_file):
    '''
    Generates musicxml file for the given midi file.

    @param midi_file Path to midi file.
    '''
    xml_file = os.path.splitext(midi_file)[0] + '.xml'
    if not os.path.exists(xml_file):
        score = music21.converter.parse(midi_file)
        score.write('musicxml', fp=xml_file)

def generate_fingering_from_musicxml(musicxml_file):
    '''
    Generates pianoplayer fingering data from the given musicXML file.
    '''
    fname = os.path.splitext(musicxml_file)[0]
    rh_fingering = fname + '_rh.txt'
    lh_fingering = fname + '_lh.txt'

    if not os.path.exists(rh_fingering) or not os.path.exists(lh_fingering) or True:
        core.run_annotate(
            musicxml_file,
            outputfile=rh_fingering,
            n_measures=800,
            depth=9,
            right_only=True,
            quiet=False)

        core.run_annotate(
            musicxml_file,
            outputfile=lh_fingering,
            n_measures=800,
            depth=9,
            left_only=True,
            quiet=False)

        print(f"done processing file {musicxml_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-midi', type=str, default=None)
    parser.add_argument('-musicxml', type=str, default=None)
    args = parser.parse_args()

    if args.midi is not None:
        midi_to_musicxml(args.midi)
    elif args.musicxml is not None:
        generate_fingering_from_musicxml(args.musicxml)
    else:
        parser.print_help()
        parser.exit(1)
