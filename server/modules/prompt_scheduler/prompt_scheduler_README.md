# Prompt Scheduler

The Prompt Scheduler is a module that allows for sequential prompt transitions in the Real-Time Latent Consistency Model. It reads prompts from a file and manages transitions between them as source/target pairs.

## How It Works

1. The Prompt Scheduler reads prompts from a file in the `prompts` directory.
2. Each line in the file represents a single prompt.
3. The scheduler transitions between prompts sequentially: A→B, B→C, C→D, etc.
4. When the prompt travel factor reaches its maximum boundary, the source prompt changes.
5. When the prompt travel factor reaches its minimum boundary, the target prompt changes.
6. This creates a smooth transition between prompts as the factor oscillates between min and max.

## Configuration

The Prompt Scheduler can be configured using the following command-line arguments:

- `--use-prompt-scheduler`: Enable the prompt scheduler
- `--prompts-dir`: Directory containing prompt files (default: "prompts")
- `--prompt-file-pattern`: Pattern to match prompt files (default: "*.txt")
- `--loop-prompts`: Loop back to the beginning when reaching the end of prompts (default: True)
- `--no-loop-prompts`: Don't loop back to the beginning when reaching the end of prompts

## Example Prompt File

Create a file in the `prompts` directory with one prompt per line:

```
a red monkey in a jungle
a blue dog in a city
a green dragon in a mountain
a yellow bird in a forest
a purple cat in a desert
```

## Integration with Prompt Travel Scheduler

The Prompt Scheduler is integrated with the Prompt Travel Scheduler, which manages the transition factor between prompts. When the Prompt Travel Scheduler is enabled and the factor reaches its boundaries, the Prompt Scheduler updates the source and target prompts accordingly.

## Usage

1. Enable the Prompt Travel Scheduler with `--use-prompt-travel-scheduler`
2. Enable the Prompt Scheduler with `--use-prompt-scheduler`
3. Create a prompt file in the `prompts` directory
4. Run the application

The system will automatically transition between prompts as the factor oscillates between min and max. 