import os
import argparse
import pianoplayer.core as core

def generate_fingering_from_musicxml(music_file):
    '''
    Generates pianoplayer fingering data from the given midi or musicXML file.
    '''
    fname = os.path.splitext(music_file)[0]
    rh_fingering = fname + '_rh.txt'
    lh_fingering = fname + '_lh.txt'

    if not os.path.exists(rh_fingering) or not os.path.exists(lh_fingering) or True:
        # Generate right fingering
        try:
            core.run_annotate(
                music_file,
                outputfile=rh_fingering,
                n_measures=800,
                depth=3,
                right_only=True,
                quiet=True)
        except IndexError as e:
            print('no right fingering data')

        # Generate left fingering
        try:
            core.run_annotate(
                music_file,
                outputfile=lh_fingering,
                n_measures=800,
                depth=3,
                left_only=True,
                quiet=True)
        except IndexError as e:
            print('no left fingering data')

        print(f"done processing file {music_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, required=True, help='Path to musicXML or midi file to generate PIG pianoplayer fingerings from')
    args = parser.parse_args()

    generate_fingering_from_musicxml(args.dir)
