"""
Fashion Style DNA Taxonomy Folder Builder

Creates hierarchical folder structure for organizing images by taxonomy:
- archetype/
  - visual_anchor/
    - attribution/
      - images (symlinks)
"""
import os
import csv
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FolderMapping:
    """Mapping from taxonomy to folder paths."""
    attribution_id: int
    attribution_name: str
    visual_anchor_id: str
    visual_anchor_name: str
    archetype_id: int
    archetype_name: str
    folder_path: str  # Relative path from output root


def sanitize_name(name: str) -> str:
    """
    Convert a name to a safe folder name.

    - Lowercase
    - Replace spaces with underscores
    - Replace special chars with safe alternatives
    - Remove consecutive underscores
    """
    # Lowercase
    result = name.lower()

    # Replace common patterns
    result = result.replace('/', '_')
    result = result.replace('&', 'and')
    result = result.replace(' ', '_')
    result = result.replace('-', '_')
    result = result.replace("'", '')
    result = result.replace('"', '')
    result = result.replace('(', '')
    result = result.replace(')', '')
    result = result.replace(',', '')
    result = result.replace('+', '_plus_')

    # Remove any other non-alphanumeric characters
    result = re.sub(r'[^a-z0-9_]', '', result)

    # Remove consecutive underscores
    result = re.sub(r'_+', '_', result)

    # Remove leading/trailing underscores
    result = result.strip('_')

    return result


def build_taxonomy_folders(
    taxonomy_csv_path: str,
    output_dir: str,
    create_dirs: bool = True,
) -> Dict[int, FolderMapping]:
    """
    Build hierarchical folder structure from taxonomy CSV.

    Args:
        taxonomy_csv_path: Path to fashion_taxonomy_dataset.csv
        output_dir: Root output directory for taxonomy images
        create_dirs: Whether to actually create the directories

    Returns:
        Dict mapping attribution_id to FolderMapping
    """
    logger.info(f"Building folder structure from {taxonomy_csv_path}")
    logger.info(f"Output directory: {output_dir}")

    mappings = {}

    with open(taxonomy_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            attribution_id = int(row['attribution_id'])
            attribution_name = row['attribution_name']
            visual_anchor_id = row['visual_anchor_id']
            visual_anchor_name = row['visual_anchor_name']
            archetype_id = int(row['archetype_id'])
            archetype_name = row['archetype_name']

            # Build folder names
            archetype_folder = f"{archetype_id}_{sanitize_name(archetype_name)}"
            anchor_folder = f"{visual_anchor_id}_{sanitize_name(visual_anchor_name)}"
            attribution_folder = f"{attribution_id:03d}_{sanitize_name(attribution_name)}"

            # Build relative path
            relative_path = os.path.join(archetype_folder, anchor_folder, attribution_folder)
            full_path = os.path.join(output_dir, relative_path)

            # Create directory if requested
            if create_dirs:
                os.makedirs(full_path, exist_ok=True)

            mapping = FolderMapping(
                attribution_id=attribution_id,
                attribution_name=attribution_name,
                visual_anchor_id=visual_anchor_id,
                visual_anchor_name=visual_anchor_name,
                archetype_id=archetype_id,
                archetype_name=archetype_name,
                folder_path=relative_path,
            )
            mappings[attribution_id] = mapping

    logger.info(f"Created mapping for {len(mappings)} attributions")

    # Log structure summary
    archetypes = {}
    for m in mappings.values():
        if m.archetype_id not in archetypes:
            archetypes[m.archetype_id] = {'name': m.archetype_name, 'anchors': set()}
        archetypes[m.archetype_id]['anchors'].add(m.visual_anchor_id)

    logger.info("Folder structure:")
    for arch_id in sorted(archetypes.keys()):
        info = archetypes[arch_id]
        logger.info(f"  {arch_id}_{sanitize_name(info['name'])}/")
        for anchor_id in sorted(info['anchors']):
            # Find anchor name
            anchor_name = None
            for m in mappings.values():
                if m.visual_anchor_id == anchor_id:
                    anchor_name = m.visual_anchor_name
                    break
            logger.info(f"    {anchor_id}_{sanitize_name(anchor_name or 'unknown')}/")

    return mappings


def get_folder_for_attribution(
    mappings: Dict[int, FolderMapping],
    attribution_id: int,
    output_dir: str,
) -> Optional[str]:
    """
    Get the full folder path for a given attribution.

    Args:
        mappings: Dict from build_taxonomy_folders
        attribution_id: Attribution ID (1-201)
        output_dir: Root output directory

    Returns:
        Full folder path or None if not found
    """
    mapping = mappings.get(attribution_id)
    if mapping:
        return os.path.join(output_dir, mapping.folder_path)
    return None


def create_image_symlink(
    source_image_path: str,
    target_folder: str,
    image_filename: Optional[str] = None,
) -> bool:
    """
    Create a symlink to an image in the target folder.

    Args:
        source_image_path: Absolute path to source image
        target_folder: Target folder to create symlink in
        image_filename: Optional filename for the symlink (defaults to source filename)

    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(source_image_path):
            logger.warning(f"Source image not found: {source_image_path}")
            return False

        if image_filename is None:
            image_filename = os.path.basename(source_image_path)

        target_path = os.path.join(target_folder, image_filename)

        # Remove existing symlink if present
        if os.path.islink(target_path):
            os.unlink(target_path)
        elif os.path.exists(target_path):
            logger.warning(f"Target exists and is not a symlink: {target_path}")
            return False

        # Create relative symlink for portability
        rel_source = os.path.relpath(source_image_path, target_folder)
        os.symlink(rel_source, target_path)
        return True

    except Exception as e:
        logger.error(f"Error creating symlink for {source_image_path}: {e}")
        return False


def count_images_in_folders(output_dir: str) -> Dict[str, int]:
    """
    Count images (symlinks or files) in each attribution folder.

    Args:
        output_dir: Root output directory

    Returns:
        Dict mapping folder path to image count
    """
    counts = {}

    for archetype_dir in os.listdir(output_dir):
        archetype_path = os.path.join(output_dir, archetype_dir)
        if not os.path.isdir(archetype_path):
            continue

        for anchor_dir in os.listdir(archetype_path):
            anchor_path = os.path.join(archetype_path, anchor_dir)
            if not os.path.isdir(anchor_path):
                continue

            for attribution_dir in os.listdir(anchor_path):
                attribution_path = os.path.join(anchor_path, attribution_dir)
                if not os.path.isdir(attribution_path):
                    continue

                # Count images
                image_count = len([
                    f for f in os.listdir(attribution_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
                ])

                relative_path = os.path.join(archetype_dir, anchor_dir, attribution_dir)
                counts[relative_path] = image_count

    return counts


def validate_folder_structure(
    output_dir: str,
    expected_attributions: int = 201,
    min_images_per_leaf: int = 20,
) -> Dict[str, any]:
    """
    Validate the created folder structure.

    Args:
        output_dir: Root output directory
        expected_attributions: Expected number of attribution folders
        min_images_per_leaf: Minimum images expected per folder

    Returns:
        Validation report
    """
    counts = count_images_in_folders(output_dir)

    total_folders = len(counts)
    folders_with_images = sum(1 for c in counts.values() if c > 0)
    folders_meeting_min = sum(1 for c in counts.values() if c >= min_images_per_leaf)
    total_images = sum(counts.values())

    # Find folders below minimum
    below_minimum = [
        (path, count) for path, count in counts.items()
        if count < min_images_per_leaf
    ]
    below_minimum.sort(key=lambda x: x[1])

    # Find empty folders
    empty_folders = [path for path, count in counts.items() if count == 0]

    return {
        'total_folders': total_folders,
        'expected_folders': expected_attributions,
        'folders_with_images': folders_with_images,
        'folders_meeting_minimum': folders_meeting_min,
        'minimum_threshold': min_images_per_leaf,
        'total_images': total_images,
        'average_images_per_folder': total_images / total_folders if total_folders > 0 else 0,
        'coverage_percent': (folders_meeting_min / expected_attributions * 100) if expected_attributions > 0 else 0,
        'empty_folders': empty_folders,
        'below_minimum': below_minimum[:20],  # Top 20 worst
        'counts_by_folder': counts,
    }


def main():
    """CLI for building and validating folder structure."""
    import argparse

    parser = argparse.ArgumentParser(description='Taxonomy Folder Builder')
    parser.add_argument('--taxonomy', type=str,
                       default='/home/ubuntu/recSys/outfitTransformer/fashion_taxonomy_dataset.csv',
                       help='Path to taxonomy CSV')
    parser.add_argument('--output', type=str,
                       default='/home/ubuntu/recSys/outfitTransformer/data/taxonomy_images',
                       help='Output directory')
    parser.add_argument('--create', action='store_true', help='Create folder structure')
    parser.add_argument('--validate', action='store_true', help='Validate existing structure')
    parser.add_argument('--min-images', type=int, default=20, help='Minimum images per folder')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.create:
        print(f"\n=== Creating Folder Structure ===")
        mappings = build_taxonomy_folders(args.taxonomy, args.output, create_dirs=True)
        print(f"\nCreated {len(mappings)} attribution folders")

    if args.validate:
        print(f"\n=== Validating Folder Structure ===")
        report = validate_folder_structure(args.output, min_images_per_leaf=args.min_images)

        print(f"\nTotal folders: {report['total_folders']} / {report['expected_folders']} expected")
        print(f"Folders with images: {report['folders_with_images']}")
        print(f"Folders meeting minimum ({args.min_images}+): {report['folders_meeting_minimum']}")
        print(f"Coverage: {report['coverage_percent']:.1f}%")
        print(f"Total images: {report['total_images']}")
        print(f"Average per folder: {report['average_images_per_folder']:.1f}")

        if report['empty_folders']:
            print(f"\nEmpty folders: {len(report['empty_folders'])}")
            for path in report['empty_folders'][:10]:
                print(f"  - {path}")

        if report['below_minimum']:
            print(f"\nFolders below minimum ({args.min_images}):")
            for path, count in report['below_minimum']:
                print(f"  - {path}: {count} images")


if __name__ == '__main__':
    main()
