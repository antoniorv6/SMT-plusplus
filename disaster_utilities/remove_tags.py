import xml.etree.ElementTree as ET
import os

def remove_direction_tags(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Function to recursively remove 'direction' elements
    def remove_directions(element):
        directions = element.findall('direction')
        for direction in directions:
            element.remove(direction)
        for child in element:
            remove_directions(child)

    # Start the removal process from the root
    remove_directions(root)

    # Generate output file name
    file_name, file_extension = os.path.splitext(xml_file)
    output_file = f"{file_name}_processed{file_extension}"

    # Write the modified XML to a new file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    return output_file

# Get input file path from user
input_file = input("Enter the path to the XML file: ")

try:
    # Process the file and get the output file path
    output_file = remove_direction_tags(input_file)
    print(f"Processing complete. Output file: {output_file}")
except FileNotFoundError:
    print("Error: The specified file was not found.")
except ET.ParseError:
    print("Error: Unable to parse the XML file. Please ensure it's a valid XML.")
except Exception as e:
    print(f"An error occurred: {str(e)}")