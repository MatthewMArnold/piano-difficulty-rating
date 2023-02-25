import os
import argparse
import music21
import pianoplayer.core as core

def truncate_midi(midi_file, new_midi_file_dir, trunc_duration):
    '''
    @param midi_file The midi file to truncate.
    @param new_midi_file_dir The directory where the new midi file will be
        placed. The new file will have the same name as the old midi file
        with '-trunc'.
    @param trunc_duration The number of measures in the truncated midi file.
    '''
    new_midi_file_name = os.path.join(new_midi_file_dir, 'trunc-' + os.path.split(midi_file)[1])

    score = music21.converter.parse(midi_file)
    trunc_score = score.measures(0, trunc_duration)
    trunc_score.write('midi', new_midi_file_name)

    print(f'original score has duration {score.duration} new score duration {trunc_score.duration}')

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
    parser.add_argument('-trunc_midi_dir', type=str, help='directory where truncated midi directory will be located', default=None)
    parser.add_argument('-trunc_duration', type=int, help='Truncated duration length, number of measures. If the number of measures is > the midi\'s number of measures, the original midi will be returned', default=None)
    parser.add_argument('--truncate', type=bool, help='truncate the midi file', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.truncate:
        if args.midi is None:
            print('midi must be provided if you want to truncate a midi')
            parser.print_usage()
            parser.exit(1)
        if args.trunc_midi_dir is None:
            print('trunc_midi_dir must be provided')
            parser.print_usage()
            parser.exit(1)
        if args.trunc_duration is None:
            print('trunc_duration must be provided')
            parser.print_usage()
            parser.exit(1)

        truncate_midi(args.midi, args.trunc_midi_dir, args.trunc_duration)
    if args.midi is not None:
        midi_to_musicxml(args.midi)
    elif args.musicxml is not None:
        generate_fingering_from_musicxml(args.musicxml)
    else:
        print('either -midi or -musicxml must be provided')
        parser.print_usage()
        parser.exit(1)
