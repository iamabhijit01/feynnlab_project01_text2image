import {
  ChakraProvider,
  Heading,
  Container,
  Text,
  Input,
  Button,
  Wrap,
  Stack, 
  Image,
  Link,
  SkeletonCircle,
  SkeletonText,
}
from "@chakra-ui/react";
//axios for api calls
import axios from "axios";
import { useState } from "react";

const App = () => {
  const [image, updateImage] = useState();
  const [prompt, updatePrompt] = useState();
  const [loading, updateLoading] = useState();

  const generate = async (prompt) => {
    updateLoading(true);
    //using POST method of our api
    const result = await axios.post(`http://127.0.0.1:8000/im_generate/?prompt=${prompt}`);
    updateImage(result.data);
    updateLoading(false);
  };

  return (
    <ChakraProvider>
      <Container>
        <Heading>Generate Image</Heading>
        <Text marginBottom={"10px"}>
          This react application leverages the model trained by Stability AI and
          Runway ML to generate images using the Stable Diffusion Deep Learning
          model. The model can be found via github here{" "}
          <Link href={"https://github.com/CompVis/stable-diffusion"}>
            Github Repo
          </Link>
        </Text>

        <Wrap marginBottom={"20px"}>
          <Input
            value={prompt}
            onChange={(e) => updatePrompt(e.target.value)}
            width={"350px"}
          ></Input>
          <Button onClick={(e) => generate(prompt)} colorScheme={"Blue"}>
            Generate
          </Button>
        </Wrap>

        {loading ? (
          <Stack>
            <SkeletonCircle />
            <SkeletonText />
          </Stack>
        ) : image ? (
          <Image src={`data:image/png;base64,${image}`}/>
        ) : null}
      </Container>
    </ChakraProvider>
  );
};

export default App;