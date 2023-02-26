'''
Truncates midi file or generates musicXML data for some provided directory.
'''

import argparse
import generate_data
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, required=True)
    parser.add_argument('--truncate', type=bool, required=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('-trunc_midi_dir', type=str, help='directory where truncated midi directory will be located', default=None)
    parser.add_argument('-trunc_duration', type=int, help='Truncated duration length, number of measures. If the number of measures is > the midi\'s number of measures, the original midi will be returned', default=None)
    args = parser.parse_args()

    print(args.dir)

    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if os.path.splitext(file)[1] == '.midi':
                fullpath = os.path.join(root, file)

                if args.truncate:
                    if args.trunc_duration is None:
                        parser.print_help()
                        parser.exit(1)
                    print(f'truncating midi {fullpath}')
                    generate_data.truncate_midi(fullpath, args.trunc_midi_dir, args.trunc_duration)
                else:
                    print(f'generating musicXML for {fullpath}')
                    generate_data.midi_to_musicxml(fullpath)
