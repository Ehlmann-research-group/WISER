from abc import abstractmethod
from typing import Union, Dict, Any
import numpy as np


class Serializable:
    """
    This class is used to serialize and deserialize objects.
    It should mainly be used for objects that are not deep copyable.
    It should be used in the context of subprocessing.
    """

    @abstractmethod
    def get_serialized_form(self) -> "SerializedForm":
        """
        This should return all of the information needed to recreate this object.
        The first element is this class, so we can get the deserialize_into_class function
        The second element is a string that represents the file path to the dataset, or a numpy array
        that represents the data in the dataset. The third element is a dictionary that represents
        the metadata needed to recreate this object.
        """
        raise NotImplementedError("This method must be implemented by the subclass")

    @staticmethod
    @abstractmethod
    def deserialize_into_class(
        serializedForm: "SerializedForm",
    ) -> "Serializable":
        """
        This should recreate the object from the serialized form that is
        obtained from the get_serialized_form method.
        """
        raise NotImplementedError("This method must be implemented by the subclass")


class SerializedForm:
    """
    This class is used to represent the serialized form of an object.
    It is used to store the serialized form of an object in a way that is easy to
    serialize and deserialize.
    """

    def __init__(
        self,
        serializable_class: "Serializable",
        serialize_value: Union[str, np.ndarray, bool, np.float32],
        metadata: Dict,
    ):
        """
        Args:
            - serializable_class: The class of the object that is being serialized. This will have
            the deserialize_into_class method.
            - serialize_value: The value that is being serialized. This can be a string that
            represents the file path to the dataset, or a numpy array
            that represents the data in the dataset.
            - metadata: A dictionary that represents the metadata needed to recreate this object.
        """
        self._serializable_class = serializable_class
        self._serialize_value = serialize_value
        self._metadata = metadata

    def get_serializable_class(self) -> "Serializable":
        return self._serializable_class

    def get_serialize_value(self) -> Union[str, np.ndarray, bool, np.float32]:
        return self._serialize_value

    def get_metadata(self) -> Dict:
        return self._metadata


class BasicValueSerialized(Serializable):
    """
    While primitives are already serializable, this class makes working
    with serialized primitives alongside our more complex serialized classes
    easier because everything will have the same interface.

    Attributes:
        primitive_value: A primitive value that we want to wrap in this class
    """

    def __init__(self, value: Any):
        self._value = value

    def get_basic_value(self):
        return self._value

    def get_serialized_form(self):
        return SerializedForm(
            serializable_class=BasicValueSerialized,
            serialize_value=self._value,
            metadata={},
        )

    @staticmethod
    def deserialize_into_class(serialize_value: str, metadata: Dict):
        return BasicValueSerialized(serialize_value)
