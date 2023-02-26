import os
import argparse
import music21

def truncate_midi(midi_file, new_midi_file_dir, trunc_duration):
    '''
    TODO it seems like music21 generated midi files cannot be read properly
    by pretty midi.
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-midi', type=str, default=None)
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
    else:
        print('either -midi or -musicxml must be provided')
        parser.print_usage()
        parser.exit(1)
