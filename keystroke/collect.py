"""Collect fixed-text timings only while typing in this application's entry."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from time import perf_counter

from .features import KEYS, PASSWORD, KeyEvent, extract_features


class Capture:
    """One sample; physical key codes match releases to their presses."""

    def __init__(self):
        self.events = []
        self.active = {}
        self.count = 0

    @property
    def complete(self):
        return self.count == len(KEYS) and not self.active

    def press(self, code, character, timestamp):
        expected = PASSWORD + '\r'
        if self.count >= len(KEYS) or character != expected[self.count] or code in self.active:
            raise ValueError('Incorrect or repeated key. Reset and type the fixed text again.')
        key = KEYS[self.count]
        self.active[code] = key
        self.events.append(KeyEvent(key, 'down', timestamp))
        self.count += 1

    def release(self, code, timestamp):
        if code in self.active:
            self.events.append(KeyEvent(self.active.pop(code), 'up', timestamp))

    def export(self, output_dir, subject, session, repetition):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', subject):
            raise ValueError('Use a short subject ID containing letters, digits, underscores, or hyphens.')
        if not 1 <= session <= 8 or not 1 <= repetition <= 50:
            raise ValueError('Session must be 1-8 and repetition must be 1-50.')
        extract_features(self.events)  # Reject incomplete samples before saving.
        destination = Path(output_dir) / f'{subject}_{session}_{repetition}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(subject=subject, sessionIndex=session, rep=repetition,
                       events=[asdict(event) for event in self.events])
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
        return destination


def main():
    import tkinter as tk
    from tkinter import ttk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/captures'))
    args = parser.parse_args()
    root = tk.Tk()
    root.title('Keystroke Timing Sample')
    frame = ttk.Frame(root, padding=20)
    frame.grid()
    ttk.Label(frame, text=f'Type the fixed sample: {PASSWORD} then press Enter.').grid(row=0, columnspan=2, pady=8)
    ttk.Label(frame, text='Only this entry is recorded. Do not type personal passwords.').grid(row=1, columnspan=2)
    variables = []
    for row, (label, initial) in enumerate([('Subject ID', 'local01'), ('Session', '1'), ('Repetition', '1')], start=2):
        variable = tk.StringVar(value=initial)
        variables.append(variable)
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w')
        ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, pady=4)
    display = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=display, width=40)
    entry.grid(row=5, columnspan=2, pady=12)
    status = tk.StringVar(value='Click the typing entry to begin. Reset after any mistake.')
    ttk.Label(frame, textvariable=status, wraplength=440).grid(row=6, columnspan=2)
    capture = Capture()
    start = perf_counter()

    def reset(message='Ready. Type the fixed sample.'):
        nonlocal capture, start
        capture, start = Capture(), perf_counter()
        display.set('')
        status.set(message)
        entry.focus_set()

    def press(event):
        if event.keysym in {'Shift_L', 'Shift_R', 'Caps_Lock'}:
            return 'break'
        try:
            character = '\r' if event.keysym == 'Return' else event.char
            capture.press(event.keycode, character, perf_counter() - start)
            display.set(PASSWORD[:capture.count])
            status.set('Release all keys, then save.' if capture.count == len(KEYS) else 'Recording sample...')
        except ValueError as error:
            reset(str(error))
        return 'break'

    def release(event):
        capture.release(event.keycode, perf_counter() - start)
        if capture.complete:
            status.set('Sample complete. Click Save sample.')
        return 'break'

    def save():
        try:
            path = capture.export(args.output_dir, variables[0].get().strip(),
                                  int(variables[1].get()), int(variables[2].get()))
            reset(f'Saved {path.name}. Change repetition before saving another sample.')
        except (ValueError, OSError) as error:
            status.set(str(error))

    entry.bind('<KeyPress>', press)
    entry.bind('<KeyRelease>', release)
    # Switching away mid-sample invalidates it because releases may be missed.
    def focus_lost(_event):
        if capture.active:
            reset('Focus changed during a key press. Sample discarded.')
    entry.bind('<FocusOut>', focus_lost)
    ttk.Button(frame, text='Reset', command=reset).grid(row=7, column=0, pady=12)
    ttk.Button(frame, text='Save sample', command=save).grid(row=7, column=1, pady=12)
    entry.focus_set()
    root.mainloop()


if __name__ == '__main__':
    main()
